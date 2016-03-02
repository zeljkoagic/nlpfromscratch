Maximize
  - 2.3423173140006 edge_0_2_0_0 - 3.920830822832 edge_0_3_0_0
   - 2.6421700161724 edge_0_5_0_0 - 3.968312082375 edge_0_6_0_0
   - 2.8882596534088 edge_0_8_0_0 - 3.7494561208644 edge_0_9_0_0
   + 2.74600328363943 edge_2_3_0_0 + 2.483517926457402 edge_2_5_0_0
   + 2.515049717567242 edge_2_6_0_0 + 1.914449604434817 edge_2_8_0_0
   + 1.772713896246427 edge_2_9_0_0 - 0.5605042522670204 edge_3_2_0_0
   + 0.2886993029836677 edge_3_5_0_0 - 0.0698236270295273 edge_3_6_0_0
   - 0.2238291755724957 edge_3_8_0_0 - 0.1199377638524251 edge_3_9_0_0
   + 0.6939489867236849 edge_5_2_0_0 + 1.250164530384506 edge_5_3_0_0
   + 2.101444798684518 edge_5_6_0_0 + 1.593967224658961 edge_5_8_0_0
   + 1.366514776382502 edge_5_9_0_0 - 1.440322295477003 edge_6_2_0_0
   - 0.9374072271616551 edge_6_3_0_0 - 0.5883161843886282 edge_6_5_0_0
   + 0.2920127943584605 edge_6_8_0_0 - 0.0907997107883289 edge_6_9_0_0
   + 0.1847530956912936 edge_8_2_0_0 + 0.2238850549951759 edge_8_3_0_0
   + 1.027227293132359 edge_8_5_0_0 + 1.246442036676938 edge_8_6_0_0
   + 1.37064642294631 edge_8_9_0_0 - 2.117682049209394 edge_9_2_0_0
   - 1.594127168237644 edge_9_3_0_0 - 1.684893032777631 edge_9_5_0_0
   - 1.18071147754991 edge_9_6_0_0 - 0.479552905303744 edge_9_8_0_0
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
