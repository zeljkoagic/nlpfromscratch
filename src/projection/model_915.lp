Maximize
  - 5.852760282996 edge_0_1_0_0 - 2.8899916780152 edge_0_2_0_0
   - 1.3499979905286 edge_0_3_0_0 - 3.0369294975318 edge_0_5_0_0
   + 0.2323209780390142 edge_1_2_0_0 - 0.1788080604792626 edge_1_3_0_0
   + 0.263509257250192 edge_1_5_0_0 - 0.4993070373145999 edge_2_1_0_0
   + 1.516675671939778 edge_2_3_0_0 + 1.1152208426529 edge_2_5_0_0
   + 0.3945841589248292 edge_3_1_0_0 + 2.675870148709916 edge_3_2_0_0
   + 2.2650261473048 edge_3_5_0_0 - 1.491595612223525 edge_5_1_0_0
   - 1.187668006843117 edge_5_2_0_0 - 0.4742564529827503 edge_5_3_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0
   = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_5_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0
   = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_5_0_0 + flow_3_2_0_0 + flow_5_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0
   = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_5_0_0 + flow_5_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0 = 5
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_5_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_5_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_5_0_0 edge_5_1_0_0 edge_5_2_0_0
 edge_5_3_0_0
End
