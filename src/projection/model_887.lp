Maximize
  - 3.6814522431988 edge_0_1_0_0 - 3.0062728040212 edge_0_2_0_0
   + 0.0032823548150416 edge_0_3_0_0 - 2.474892258888 edge_0_5_0_0
   - 2.573731640088 edge_0_8_0_0 - 0.341301494074627 edge_1_2_0_0
   - 0.3432148055628917 edge_1_3_0_0 - 0.5303115957177312 edge_1_4_0_0
   - 0.8652864414869512 edge_1_5_0_0 + 0.0595151753985397 edge_1_6_0_0
   - 0.1421606584239066 edge_1_8_0_0 - 0.5945482601607118 edge_1_11_0_0
   - 0.4539732208346016 edge_1_12_0_0 - 0.5402585265725116 edge_2_1_0_0
   + 0.4151433577509008 edge_2_3_0_0 + 0.0844599589859312 edge_2_4_0_0
   + 1.192892726759167 edge_3_1_0_0 + 1.306010084107488 edge_3_2_0_0
   + 1.709804326049828 edge_3_4_0_0 - 0.4762119006778543 edge_4_1_0_0
   - 0.4590093605948593 edge_4_2_0_0 - 0.329585831725417 edge_4_6_0_0
   - 0.2137894441584084 edge_4_11_0_0 - 0.3576144144169245 edge_5_1_0_0
   + 1.124982493199284 edge_5_4_0_0 + 0.6646312615317999 edge_5_6_0_0
   + 0.1636500174056512 edge_5_8_0_0 - 0.0679636842792986 edge_5_11_0_0
   - 0.0790629272364002 edge_5_12_0_0 - 0.5924418309203426 edge_6_5_0_0
   - 0.3167955265767394 edge_6_11_0_0 - 0.2282420309109708 edge_8_1_0_0
   - 0.00221290787191932 edge_8_5_0_0 - 0.1825836381763202 edge_8_6_0_0
   - 0.0826792124071802 edge_8_11_0_0 - 0.2725033891026693 edge_8_12_0_0
   - 0.4709396436755195 edge_11_1_0_0 + 0.2283041257301045 edge_11_4_0_0
   - 0.0933230969507179 edge_11_5_0_0 - 0.2289988828654716 edge_11_6_0_0
   - 0.2677277522487262 edge_11_8_0_0 + 0.2172114999679033 edge_11_12_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_8_1_0_0 + edge_11_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_8_0_0
   - flow_1_11_0_0 - flow_1_12_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_5_1_0_0 + flow_8_1_0_0 + flow_11_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 + flow_3_2_0_0 + flow_4_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 = 1
 one_parent_4: edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_1_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_6_0_0 - flow_4_11_0_0
   + flow_5_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_6_5_0_0 + edge_8_5_0_0
   + edge_11_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 - flow_5_1_0_0
   - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_8_0_0 - flow_5_11_0_0
   - flow_5_12_0_0 + flow_6_5_0_0 + flow_8_5_0_0 + flow_11_5_0_0 = 1
 one_parent_6: edge_1_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0 + edge_8_6_0_0
   + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_1_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0
   - flow_6_5_0_0 - flow_6_11_0_0 + flow_8_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_5_8_0_0 + edge_11_8_0_0
   = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_5_8_0_0
   - flow_8_1_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_11_0_0
   - flow_8_12_0_0 + flow_11_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_1_11_0_0 + edge_4_11_0_0 + edge_5_11_0_0
   + edge_6_11_0_0 + edge_8_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_4_11_0_0 + flow_5_11_0_0
   + flow_6_11_0_0 + flow_8_11_0_0 - flow_11_1_0_0 - flow_11_4_0_0
   - flow_11_5_0_0 - flow_11_6_0_0 - flow_11_8_0_0 - flow_11_12_0_0 = 1
 one_parent_12: edge_1_12_0_0 + edge_5_12_0_0 + edge_8_12_0_0
   + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_1_12_0_0 + flow_5_12_0_0 + flow_8_12_0_0
   + flow_11_12_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0
   + flow_0_8_0_0 = 12
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_8_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_5_0_0 edge_1_6_0_0
 edge_1_8_0_0 edge_1_11_0_0 edge_1_12_0_0 edge_2_1_0_0 edge_2_3_0_0
 edge_2_4_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_6_0_0 edge_4_11_0_0 edge_5_1_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_8_0_0 edge_5_11_0_0 edge_5_12_0_0 edge_6_5_0_0
 edge_6_11_0_0 edge_8_1_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_11_0_0
 edge_8_12_0_0 edge_11_1_0_0 edge_11_4_0_0 edge_11_5_0_0 edge_11_6_0_0
 edge_11_8_0_0 edge_11_12_0_0
End
