Maximize
  - 3.6220187401393 edge_0_1_0_0 - 2.2477597302688 edge_0_2_0_0
   - 0.38175223591587 edge_0_3_0_0 - 2.6169206967808 edge_0_8_0_0
   - 2.180929835034 edge_0_29_0_0 - 0.1351884224347374 edge_1_2_0_0
   - 0.2300842819978665 edge_1_3_0_0 - 0.4154817405493774 edge_1_7_0_0
   - 0.8810621187863015 edge_1_8_0_0 - 0.8815798495480351 edge_1_29_0_0
   + 0.021203988987361 edge_2_1_0_0 + 0.6452385676056426 edge_2_3_0_0
   - 0.0288196888106519 edge_2_7_0_0 + 0.1969043669250731 edge_2_8_0_0
   + 0.1372762186456518 edge_2_29_0_0 + 1.369225532429451 edge_3_1_0_0
   + 1.627601980550739 edge_3_2_0_0 + 0.6303471528716689 edge_3_7_0_0
   + 2.260641727348621 edge_3_8_0_0 + 2.022844223689463 edge_3_29_0_0
   - 0.4561021920673486 edge_7_1_0_0 - 0.4476263010676874 edge_7_2_0_0
   + 0.5078412560640747 edge_7_8_0_0 + 0.1703166590944994 edge_7_29_0_0
   - 0.6935235780230045 edge_8_1_0_0 - 0.0695719147725212 edge_8_2_0_0
   + 0.091006742971681 edge_8_3_0_0 + 2.094603246418814 edge_8_7_0_0
   + 0.4292127760374715 edge_8_29_0_0 - 0.2658488754420853 edge_29_1_0_0
   + 0.2629680549674296 edge_29_2_0_0 + 0.4481279181701951 edge_29_3_0_0
   + 1.000551648297623 edge_29_7_0_0 + 2.499737176140775 edge_29_8_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_7_1_0_0
   + edge_8_1_0_0 + edge_29_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_7_0_0 - flow_1_8_0_0 - flow_1_29_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0 + flow_29_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_7_2_0_0
   + edge_8_2_0_0 + edge_29_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_29_0_0
   + flow_3_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 + flow_29_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_8_3_0_0
   + edge_29_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_29_0_0 + flow_8_3_0_0 + flow_29_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_8_7_0_0
   + edge_29_7_0_0 = 1
 consume_one_flow_unit_7: flow_1_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_8_0_0 - flow_7_29_0_0
   + flow_8_7_0_0 + flow_29_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_7_8_0_0 + edge_29_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_7_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0
   - flow_8_3_0_0 - flow_8_7_0_0 - flow_8_29_0_0 + flow_29_8_0_0 = 1
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
 one_parent_18: = 1
 consume_one_flow_unit_18: = 1
 one_parent_19: = 1
 consume_one_flow_unit_19: = 1
 one_parent_20: = 1
 consume_one_flow_unit_20: = 1
 one_parent_21: = 1
 consume_one_flow_unit_21: = 1
 one_parent_22: = 1
 consume_one_flow_unit_22: = 1
 one_parent_23: = 1
 consume_one_flow_unit_23: = 1
 one_parent_24: = 1
 consume_one_flow_unit_24: = 1
 one_parent_25: = 1
 consume_one_flow_unit_25: = 1
 one_parent_26: = 1
 consume_one_flow_unit_26: = 1
 one_parent_27: = 1
 consume_one_flow_unit_27: = 1
 one_parent_28: = 1
 consume_one_flow_unit_28: = 1
 one_parent_29: edge_0_29_0_0 + edge_1_29_0_0 + edge_2_29_0_0
   + edge_3_29_0_0 + edge_7_29_0_0 + edge_8_29_0_0 = 1
 consume_one_flow_unit_29: flow_0_29_0_0 + flow_1_29_0_0 + flow_2_29_0_0
   + flow_3_29_0_0 + flow_7_29_0_0 + flow_8_29_0_0 - flow_29_1_0_0
   - flow_29_2_0_0 - flow_29_3_0_0 - flow_29_7_0_0 - flow_29_8_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_8_0_0
   + flow_0_29_0_0 = 29
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_29_0_0: flow_0_29_0_0 - 1000 edge_0_29_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_29_0_0: flow_1_29_0_0 - 1000 edge_1_29_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_29_0_0: flow_2_29_0_0 - 1000 edge_2_29_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_29_0_0: flow_3_29_0_0 - 1000 edge_3_29_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_29_0_0: flow_7_29_0_0 - 1000 edge_7_29_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_29_0_0: flow_8_29_0_0 - 1000 edge_8_29_0_0 <= 0
 inactive_no_flow_29_1_0_0: flow_29_1_0_0 - 1000 edge_29_1_0_0 <= 0
 inactive_no_flow_29_2_0_0: flow_29_2_0_0 - 1000 edge_29_2_0_0 <= 0
 inactive_no_flow_29_3_0_0: flow_29_3_0_0 - 1000 edge_29_3_0_0 <= 0
 inactive_no_flow_29_7_0_0: flow_29_7_0_0 - 1000 edge_29_7_0_0 <= 0
 inactive_no_flow_29_8_0_0: flow_29_8_0_0 - 1000 edge_29_8_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_8_0_0 edge_0_29_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_1_29_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_29_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_29_0_0
 edge_7_1_0_0 edge_7_2_0_0 edge_7_8_0_0 edge_7_29_0_0 edge_8_1_0_0
 edge_8_2_0_0 edge_8_3_0_0 edge_8_7_0_0 edge_8_29_0_0 edge_29_1_0_0
 edge_29_2_0_0 edge_29_3_0_0 edge_29_7_0_0 edge_29_8_0_0
End
