Maximize
  - 2.7417914169546 edge_0_1_0_0 - 3.0232569710222 edge_0_4_0_0
   - 2.7377574220994 edge_0_6_0_0 - 3.453923725450299 edge_0_18_0_0
   - 0.4944078702806846 edge_1_6_0_0 + 0.2366960732077588 edge_1_18_0_0
   - 0.8922839833009617 edge_4_1_0_0 + 0.6957737430908749 edge_4_6_0_0
   + 0.0781600732487667 edge_4_18_0_0 - 0.00439932706239545 edge_6_1_0_0
   + 3.411218758874084 edge_6_4_0_0 + 1.369538656758206 edge_6_18_0_0
   - 1.197835565131895 edge_18_1_0_0 - 0.99975229194044 edge_18_6_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_4_1_0_0 + edge_6_1_0_0 + edge_18_1_0_0
   = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_6_0_0 - flow_1_18_0_0
   + flow_4_1_0_0 + flow_6_1_0_0 + flow_18_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: edge_0_4_0_0 + edge_6_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 - flow_4_1_0_0 - flow_4_6_0_0
   - flow_4_18_0_0 + flow_6_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_4_6_0_0 + edge_18_6_0_0
   = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_4_6_0_0
   - flow_6_1_0_0 - flow_6_4_0_0 - flow_6_18_0_0 + flow_18_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: = 1
 consume_one_flow_unit_14: = 1
 one_parent_15: = 1
 consume_one_flow_unit_15: = 1
 one_parent_16: = 1
 consume_one_flow_unit_16: = 1
 one_parent_17: = 1
 consume_one_flow_unit_17: = 1
 one_parent_18: edge_0_18_0_0 + edge_1_18_0_0 + edge_4_18_0_0
   + edge_6_18_0_0 = 1
 consume_one_flow_unit_18: flow_0_18_0_0 + flow_1_18_0_0 + flow_4_18_0_0
   + flow_6_18_0_0 - flow_18_1_0_0 - flow_18_6_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_4_0_0 + flow_0_6_0_0 + flow_0_18_0_0
   = 18
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_18_0_0: flow_0_18_0_0 - 1000 edge_0_18_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_18_0_0: flow_1_18_0_0 - 1000 edge_1_18_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_18_0_0: flow_4_18_0_0 - 1000 edge_4_18_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_18_0_0: flow_6_18_0_0 - 1000 edge_6_18_0_0 <= 0
 inactive_no_flow_18_1_0_0: flow_18_1_0_0 - 1000 edge_18_1_0_0 <= 0
 inactive_no_flow_18_6_0_0: flow_18_6_0_0 - 1000 edge_18_6_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_0_18_0_0 edge_1_6_0_0
 edge_1_18_0_0 edge_4_1_0_0 edge_4_6_0_0 edge_4_18_0_0 edge_6_1_0_0
 edge_6_4_0_0 edge_6_18_0_0 edge_18_1_0_0 edge_18_6_0_0
End
