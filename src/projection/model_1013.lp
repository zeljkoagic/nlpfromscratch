Maximize
  - 5.2177081216244 edge_0_3_0_0 - 4.117226779302 edge_0_4_0_0
   - 4.94977 edge_0_5_0_0 - 2.8128733815996 edge_0_6_0_0
   - 2.876293457852 edge_0_7_0_0 - 4.262134378420501 edge_0_8_0_0
   - 3.8823697670578 edge_0_10_0_0 - 3.75694 edge_0_11_0_0
   - 0.1953578097030084 edge_2_4_0_0 + 3.68751136440603 edge_3_2_0_0
   + 2.787004840526696 edge_3_4_0_0 + 1.694649389926 edge_3_5_0_0
   + 0.953022429186562 edge_3_6_0_0 - 0.5728293086433747 edge_3_7_0_0
   - 0.6789358953507397 edge_3_8_0_0 - 0.229984903406305 edge_3_10_0_0
   - 0.32656688243588 edge_3_11_0_0 + 1.1220695960548 edge_3_12_0_0
   + 1.529181994125182 edge_4_2_0_0 + 1.203936360998354 edge_4_3_0_0
   + 1.49060459289 edge_4_5_0_0 + 0.8438520790379409 edge_4_6_0_0
   - 0.0734830812146159 edge_4_7_0_0 + 0.5112312274201172 edge_4_8_0_0
   + 0.1952037292786356 edge_4_10_0_0 + 0.4899346959221401 edge_4_11_0_0
   + 1.161902044716 edge_4_12_0_0 + 1.1161174315887 edge_5_2_0_0
   + 0.10468896231196 edge_5_3_0_0 + 1.0802810744838 edge_5_4_0_0
   + 1.1026168475454 edge_5_6_0_0 - 0.3348276124448 edge_5_7_0_0
   + 0.162557878592 edge_5_8_0_0 - 0.1914799885112 edge_5_10_0_0
   - 0.121847 edge_5_11_0_0 + 0.418852 edge_5_12_0_0
   - 0.1525358595662222 edge_6_2_0_0 + 1.087307471779969 edge_6_3_0_0
   - 0.0188568208999112 edge_6_4_0_0 + 2.451387364173 edge_6_5_0_0
   + 0.3685960648273621 edge_6_7_0_0 + 1.738581964524474 edge_6_8_0_0
   + 2.630515322194471 edge_6_10_0_0 + 2.3513739994998 edge_6_11_0_0
   + 2.4365783466684 edge_6_12_0_0 - 0.3719989973187128 edge_7_3_0_0
   - 0.1604860603409115 edge_7_4_0_0 - 0.2101890296016 edge_7_5_0_0
   - 0.157939247030416 edge_7_6_0_0 + 0.5072064171020771 edge_7_8_0_0
   + 0.714529706798215 edge_7_10_0_0 + 0.3650581789676 edge_7_11_0_0
   - 0.4725181920156001 edge_7_12_0_0 - 1.028486207819832 edge_8_3_0_0
   - 1.102103574415461 edge_8_4_0_0 - 0.57898544414135 edge_8_5_0_0
   - 0.1233912033014428 edge_8_6_0_0 + 0.6098072597294075 edge_8_7_0_0
   + 0.0518662200249816 edge_8_10_0_0 - 0.414514590514 edge_8_11_0_0
   - 0.4349283241113 edge_8_12_0_0 - 0.1774469000215453 edge_10_2_0_0
   - 0.1909369198064641 edge_10_3_0_0 - 0.1583328798345067 edge_10_4_0_0
   + 0.47877397127356 edge_10_5_0_0 + 0.7056271432492088 edge_10_6_0_0
   + 1.464588861086663 edge_10_7_0_0 + 3.841549636627009 edge_10_8_0_0
   + 2.7158598370484 edge_10_11_0_0 + 0.75811595451304 edge_10_12_0_0
   - 0.14949899274161 edge_11_2_0_0 - 0.099926264026532 edge_11_3_0_0
   + 0.24588329774946 edge_11_4_0_0 + 0.946473 edge_11_5_0_0
   + 0.7085549913433199 edge_11_6_0_0 + 0.9697342317940001 edge_11_7_0_0
   + 2.1338821524745 edge_11_8_0_0 + 1.9513898829166 edge_11_10_0_0
   + 0.499006 edge_11_12_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: - flow_2_4_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_10_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0
   + edge_7_3_0_0 + edge_8_3_0_0 + edge_10_3_0_0 + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 - flow_3_2_0_0 - flow_3_4_0_0
   - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_12_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0
   + flow_10_3_0_0 + flow_11_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 + edge_10_4_0_0
   + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_10_0_0 - flow_4_11_0_0
   - flow_4_12_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0
   + flow_8_4_0_0 + flow_10_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 + edge_10_5_0_0 + edge_11_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_3_5_0_0 + flow_4_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0
   - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_10_0_0 - flow_5_11_0_0
   - flow_5_12_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0
   + flow_10_5_0_0 + flow_11_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_10_6_0_0 + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_3_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 - flow_6_12_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_10_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 + edge_10_7_0_0 + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_3_7_0_0 + flow_4_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_10_0_0
   - flow_7_11_0_0 - flow_7_12_0_0 + flow_8_7_0_0 + flow_10_7_0_0
   + flow_11_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0
   - flow_8_10_0_0 - flow_8_11_0_0 - flow_8_12_0_0 + flow_10_8_0_0
   + flow_11_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_0_10_0_0 + edge_3_10_0_0 + edge_4_10_0_0
   + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_3_10_0_0 + flow_4_10_0_0
   + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0
   - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_5_0_0
   - flow_10_6_0_0 - flow_10_7_0_0 - flow_10_8_0_0 - flow_10_11_0_0
   - flow_10_12_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_5_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0 + edge_8_11_0_0
   + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_5_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0 + flow_8_11_0_0
   + flow_10_11_0_0 - flow_11_2_0_0 - flow_11_3_0_0 - flow_11_4_0_0
   - flow_11_5_0_0 - flow_11_6_0_0 - flow_11_7_0_0 - flow_11_8_0_0
   - flow_11_10_0_0 - flow_11_12_0_0 = 1
 one_parent_12: edge_3_12_0_0 + edge_4_12_0_0 + edge_5_12_0_0
   + edge_6_12_0_0 + edge_7_12_0_0 + edge_8_12_0_0 + edge_10_12_0_0
   + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_3_12_0_0 + flow_4_12_0_0 + flow_5_12_0_0
   + flow_6_12_0_0 + flow_7_12_0_0 + flow_8_12_0_0 + flow_10_12_0_0
   + flow_11_12_0_0 = 1
 root_flow: flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 12
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
Bounds
Binaries
 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_2_4_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_10_0_0 edge_3_11_0_0 edge_3_12_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_10_0_0
 edge_4_11_0_0 edge_4_12_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_10_0_0 edge_5_11_0_0
 edge_5_12_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_6_12_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0
 edge_7_10_0_0 edge_7_11_0_0 edge_7_12_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_10_0_0 edge_8_11_0_0
 edge_8_12_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_5_0_0
 edge_10_6_0_0 edge_10_7_0_0 edge_10_8_0_0 edge_10_11_0_0 edge_10_12_0_0
 edge_11_2_0_0 edge_11_3_0_0 edge_11_4_0_0 edge_11_5_0_0 edge_11_6_0_0
 edge_11_7_0_0 edge_11_8_0_0 edge_11_10_0_0 edge_11_12_0_0
End
