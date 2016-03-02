Maximize
  - 2.134920313362 edge_0_1_0_0 + 0.022230131546628 edge_0_14_0_0
   - 0.8224521393547201 edge_0_17_0_0 - 3.2358455572605 edge_0_20_0_0
   + 0.2263915697222161 edge_1_14_0_0 + 0.1109629252346925 edge_1_17_0_0
   + 0.0910447491691196 edge_1_20_0_0 - 0.1389372215639947 edge_1_23_0_0
   + 0.6361897276538171 edge_14_1_0_0 + 0.9657372195070993 edge_14_17_0_0
   + 0.8309403605071518 edge_14_20_0_0 + 1.14265128308301 edge_14_23_0_0
   + 0.2032087606605478 edge_17_1_0_0 + 0.3892104612709021 edge_17_14_0_0
   + 2.377993529315152 edge_17_20_0_0 + 3.014994724244527 edge_17_23_0_0
   - 0.0351822754781975 edge_20_1_0_0 + 0.0212757971909507 edge_20_14_0_0
   + 0.4419394870725454 edge_20_17_0_0 - 0.1229116046323554 edge_20_23_0_0
   - 1.538412934355759 edge_23_17_0_0 - 2.217689856443759 edge_23_20_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_14_1_0_0 + edge_17_1_0_0 + edge_20_1_0_0
   = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_14_0_0 - flow_1_17_0_0
   - flow_1_20_0_0 - flow_1_23_0_0 + flow_14_1_0_0 + flow_17_1_0_0
   + flow_20_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
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
 one_parent_14: edge_0_14_0_0 + edge_1_14_0_0 + edge_17_14_0_0
   + edge_20_14_0_0 = 1
 consume_one_flow_unit_14: flow_0_14_0_0 + flow_1_14_0_0 - flow_14_1_0_0
   - flow_14_17_0_0 - flow_14_20_0_0 - flow_14_23_0_0 + flow_17_14_0_0
   + flow_20_14_0_0 = 1
 one_parent_15: = 1
 consume_one_flow_unit_15: = 1
 one_parent_16: = 1
 consume_one_flow_unit_16: = 1
 one_parent_17: edge_0_17_0_0 + edge_1_17_0_0 + edge_14_17_0_0
   + edge_20_17_0_0 + edge_23_17_0_0 = 1
 consume_one_flow_unit_17: flow_0_17_0_0 + flow_1_17_0_0 + flow_14_17_0_0
   - flow_17_1_0_0 - flow_17_14_0_0 - flow_17_20_0_0 - flow_17_23_0_0
   + flow_20_17_0_0 + flow_23_17_0_0 = 1
 one_parent_18: = 1
 consume_one_flow_unit_18: = 1
 one_parent_19: = 1
 consume_one_flow_unit_19: = 1
 one_parent_20: edge_0_20_0_0 + edge_1_20_0_0 + edge_14_20_0_0
   + edge_17_20_0_0 + edge_23_20_0_0 = 1
 consume_one_flow_unit_20: flow_0_20_0_0 + flow_1_20_0_0 + flow_14_20_0_0
   + flow_17_20_0_0 - flow_20_1_0_0 - flow_20_14_0_0 - flow_20_17_0_0
   - flow_20_23_0_0 + flow_23_20_0_0 = 1
 one_parent_21: = 1
 consume_one_flow_unit_21: = 1
 one_parent_22: = 1
 consume_one_flow_unit_22: = 1
 one_parent_23: edge_1_23_0_0 + edge_14_23_0_0 + edge_17_23_0_0
   + edge_20_23_0_0 = 1
 consume_one_flow_unit_23: flow_1_23_0_0 + flow_14_23_0_0 + flow_17_23_0_0
   + flow_20_23_0_0 - flow_23_17_0_0 - flow_23_20_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_14_0_0 + flow_0_17_0_0 + flow_0_20_0_0
   = 23
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_14_0_0: flow_0_14_0_0 - 1000 edge_0_14_0_0 <= 0
 inactive_no_flow_0_17_0_0: flow_0_17_0_0 - 1000 edge_0_17_0_0 <= 0
 inactive_no_flow_0_20_0_0: flow_0_20_0_0 - 1000 edge_0_20_0_0 <= 0
 inactive_no_flow_1_14_0_0: flow_1_14_0_0 - 1000 edge_1_14_0_0 <= 0
 inactive_no_flow_1_17_0_0: flow_1_17_0_0 - 1000 edge_1_17_0_0 <= 0
 inactive_no_flow_1_20_0_0: flow_1_20_0_0 - 1000 edge_1_20_0_0 <= 0
 inactive_no_flow_1_23_0_0: flow_1_23_0_0 - 1000 edge_1_23_0_0 <= 0
 inactive_no_flow_14_1_0_0: flow_14_1_0_0 - 1000 edge_14_1_0_0 <= 0
 inactive_no_flow_14_17_0_0: flow_14_17_0_0 - 1000 edge_14_17_0_0 <= 0
 inactive_no_flow_14_20_0_0: flow_14_20_0_0 - 1000 edge_14_20_0_0 <= 0
 inactive_no_flow_14_23_0_0: flow_14_23_0_0 - 1000 edge_14_23_0_0 <= 0
 inactive_no_flow_17_1_0_0: flow_17_1_0_0 - 1000 edge_17_1_0_0 <= 0
 inactive_no_flow_17_14_0_0: flow_17_14_0_0 - 1000 edge_17_14_0_0 <= 0
 inactive_no_flow_17_20_0_0: flow_17_20_0_0 - 1000 edge_17_20_0_0 <= 0
 inactive_no_flow_17_23_0_0: flow_17_23_0_0 - 1000 edge_17_23_0_0 <= 0
 inactive_no_flow_20_1_0_0: flow_20_1_0_0 - 1000 edge_20_1_0_0 <= 0
 inactive_no_flow_20_14_0_0: flow_20_14_0_0 - 1000 edge_20_14_0_0 <= 0
 inactive_no_flow_20_17_0_0: flow_20_17_0_0 - 1000 edge_20_17_0_0 <= 0
 inactive_no_flow_20_23_0_0: flow_20_23_0_0 - 1000 edge_20_23_0_0 <= 0
 inactive_no_flow_23_17_0_0: flow_23_17_0_0 - 1000 edge_23_17_0_0 <= 0
 inactive_no_flow_23_20_0_0: flow_23_20_0_0 - 1000 edge_23_20_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_14_0_0 edge_0_17_0_0 edge_0_20_0_0 edge_1_14_0_0
 edge_1_17_0_0 edge_1_20_0_0 edge_1_23_0_0 edge_14_1_0_0 edge_14_17_0_0
 edge_14_20_0_0 edge_14_23_0_0 edge_17_1_0_0 edge_17_14_0_0 edge_17_20_0_0
 edge_17_23_0_0 edge_20_1_0_0 edge_20_14_0_0 edge_20_17_0_0 edge_20_23_0_0
 edge_23_17_0_0 edge_23_20_0_0
End
