Maximize
  - 1.4435434864872 edge_0_2_0_0 - 1.4489001603832 edge_0_22_0_0
   + 0.00418098874566499 edge_1_2_0_0 + 0.0741690037416893 edge_2_1_0_0
   + 0.3162466430821891 edge_2_3_0_0 + 0.1769342027596703 edge_2_9_0_0
   + 0.3139608318354157 edge_2_13_0_0 - 0.0542991820080395 edge_2_20_0_0
   + 0.1378627849017503 edge_2_21_0_0 + 0.1882491279590601 edge_2_22_0_0
   + 0.9174800600510427 edge_2_26_0_0 - 0.2029309360637687 edge_3_13_0_0
   - 0.1060834427729337 edge_9_1_0_0 + 0.0477260671747836 edge_9_3_0_0
   - 0.1940866098377415 edge_9_13_0_0 - 0.1641736029904854 edge_9_22_0_0
   - 0.1524125667909761 edge_13_3_0_0 - 0.2599500194693621 edge_13_20_0_0
   + 0.0129290237861608 edge_13_21_0_0 - 0.1454937087211657 edge_13_22_0_0
   - 0.1955555995544791 edge_20_13_0_0 - 0.0724391833042605 edge_20_21_0_0
   - 0.1110542597369802 edge_20_22_0_0 - 0.1384626083817968 edge_21_1_0_0
   - 0.1662493935734715 edge_21_2_0_0 - 0.0267979459821474 edge_21_9_0_0
   + 0.1283373392328913 edge_21_13_0_0 + 0.0691893036608296 edge_21_20_0_0
   - 0.0745811642597612 edge_22_1_0_0 - 0.0442250644622345 edge_22_2_0_0
   + 0.00924583393014163 edge_22_3_0_0 + 0.0706582610010615 edge_22_9_0_0
   + 0.3407553325196685 edge_22_13_0_0 + 0.8248726324759023 edge_22_20_0_0
   + 0.510493151195855 edge_22_21_0_0 + 0.6327288790609149 edge_22_26_0_0
Subject To
 one_parent_1: edge_2_1_0_0 + edge_9_1_0_0 + edge_21_1_0_0 + edge_22_1_0_0
   = 1
 consume_one_flow_unit_1: - flow_1_2_0_0 + flow_2_1_0_0 + flow_9_1_0_0
   + flow_21_1_0_0 + flow_22_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_21_2_0_0 + edge_22_2_0_0
   = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_9_0_0 - flow_2_13_0_0 - flow_2_20_0_0
   - flow_2_21_0_0 - flow_2_22_0_0 - flow_2_26_0_0 + flow_21_2_0_0
   + flow_22_2_0_0 = 1
 one_parent_3: edge_2_3_0_0 + edge_9_3_0_0 + edge_13_3_0_0 + edge_22_3_0_0
   = 1
 consume_one_flow_unit_3: flow_2_3_0_0 - flow_3_13_0_0 + flow_9_3_0_0
   + flow_13_3_0_0 + flow_22_3_0_0 = 1
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
 one_parent_9: edge_2_9_0_0 + edge_21_9_0_0 + edge_22_9_0_0 = 1
 consume_one_flow_unit_9: flow_2_9_0_0 - flow_9_1_0_0 - flow_9_3_0_0
   - flow_9_13_0_0 - flow_9_22_0_0 + flow_21_9_0_0 + flow_22_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: edge_2_13_0_0 + edge_3_13_0_0 + edge_9_13_0_0
   + edge_20_13_0_0 + edge_21_13_0_0 + edge_22_13_0_0 = 1
 consume_one_flow_unit_13: flow_2_13_0_0 + flow_3_13_0_0 + flow_9_13_0_0
   - flow_13_3_0_0 - flow_13_20_0_0 - flow_13_21_0_0 - flow_13_22_0_0
   + flow_20_13_0_0 + flow_21_13_0_0 + flow_22_13_0_0 = 1
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
 one_parent_20: edge_2_20_0_0 + edge_13_20_0_0 + edge_21_20_0_0
   + edge_22_20_0_0 = 1
 consume_one_flow_unit_20: flow_2_20_0_0 + flow_13_20_0_0 - flow_20_13_0_0
   - flow_20_21_0_0 - flow_20_22_0_0 + flow_21_20_0_0 + flow_22_20_0_0 = 1
 one_parent_21: edge_2_21_0_0 + edge_13_21_0_0 + edge_20_21_0_0
   + edge_22_21_0_0 = 1
 consume_one_flow_unit_21: flow_2_21_0_0 + flow_13_21_0_0 + flow_20_21_0_0
   - flow_21_1_0_0 - flow_21_2_0_0 - flow_21_9_0_0 - flow_21_13_0_0
   - flow_21_20_0_0 + flow_22_21_0_0 = 1
 one_parent_22: edge_0_22_0_0 + edge_2_22_0_0 + edge_9_22_0_0
   + edge_13_22_0_0 + edge_20_22_0_0 = 1
 consume_one_flow_unit_22: flow_0_22_0_0 + flow_2_22_0_0 + flow_9_22_0_0
   + flow_13_22_0_0 + flow_20_22_0_0 - flow_22_1_0_0 - flow_22_2_0_0
   - flow_22_3_0_0 - flow_22_9_0_0 - flow_22_13_0_0 - flow_22_20_0_0
   - flow_22_21_0_0 - flow_22_26_0_0 = 1
 one_parent_23: = 1
 consume_one_flow_unit_23: = 1
 one_parent_24: = 1
 consume_one_flow_unit_24: = 1
 one_parent_25: = 1
 consume_one_flow_unit_25: = 1
 one_parent_26: edge_2_26_0_0 + edge_22_26_0_0 = 1
 consume_one_flow_unit_26: flow_2_26_0_0 + flow_22_26_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_22_0_0 = 26
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_22_0_0: flow_0_22_0_0 - 1000 edge_0_22_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_2_20_0_0: flow_2_20_0_0 - 1000 edge_2_20_0_0 <= 0
 inactive_no_flow_2_21_0_0: flow_2_21_0_0 - 1000 edge_2_21_0_0 <= 0
 inactive_no_flow_2_22_0_0: flow_2_22_0_0 - 1000 edge_2_22_0_0 <= 0
 inactive_no_flow_2_26_0_0: flow_2_26_0_0 - 1000 edge_2_26_0_0 <= 0
 inactive_no_flow_3_13_0_0: flow_3_13_0_0 - 1000 edge_3_13_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_13_0_0: flow_9_13_0_0 - 1000 edge_9_13_0_0 <= 0
 inactive_no_flow_9_22_0_0: flow_9_22_0_0 - 1000 edge_9_22_0_0 <= 0
 inactive_no_flow_13_3_0_0: flow_13_3_0_0 - 1000 edge_13_3_0_0 <= 0
 inactive_no_flow_13_20_0_0: flow_13_20_0_0 - 1000 edge_13_20_0_0 <= 0
 inactive_no_flow_13_21_0_0: flow_13_21_0_0 - 1000 edge_13_21_0_0 <= 0
 inactive_no_flow_13_22_0_0: flow_13_22_0_0 - 1000 edge_13_22_0_0 <= 0
 inactive_no_flow_20_13_0_0: flow_20_13_0_0 - 1000 edge_20_13_0_0 <= 0
 inactive_no_flow_20_21_0_0: flow_20_21_0_0 - 1000 edge_20_21_0_0 <= 0
 inactive_no_flow_20_22_0_0: flow_20_22_0_0 - 1000 edge_20_22_0_0 <= 0
 inactive_no_flow_21_1_0_0: flow_21_1_0_0 - 1000 edge_21_1_0_0 <= 0
 inactive_no_flow_21_2_0_0: flow_21_2_0_0 - 1000 edge_21_2_0_0 <= 0
 inactive_no_flow_21_9_0_0: flow_21_9_0_0 - 1000 edge_21_9_0_0 <= 0
 inactive_no_flow_21_13_0_0: flow_21_13_0_0 - 1000 edge_21_13_0_0 <= 0
 inactive_no_flow_21_20_0_0: flow_21_20_0_0 - 1000 edge_21_20_0_0 <= 0
 inactive_no_flow_22_1_0_0: flow_22_1_0_0 - 1000 edge_22_1_0_0 <= 0
 inactive_no_flow_22_2_0_0: flow_22_2_0_0 - 1000 edge_22_2_0_0 <= 0
 inactive_no_flow_22_3_0_0: flow_22_3_0_0 - 1000 edge_22_3_0_0 <= 0
 inactive_no_flow_22_9_0_0: flow_22_9_0_0 - 1000 edge_22_9_0_0 <= 0
 inactive_no_flow_22_13_0_0: flow_22_13_0_0 - 1000 edge_22_13_0_0 <= 0
 inactive_no_flow_22_20_0_0: flow_22_20_0_0 - 1000 edge_22_20_0_0 <= 0
 inactive_no_flow_22_21_0_0: flow_22_21_0_0 - 1000 edge_22_21_0_0 <= 0
 inactive_no_flow_22_26_0_0: flow_22_26_0_0 - 1000 edge_22_26_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_22_0_0 edge_1_2_0_0 edge_2_1_0_0 edge_2_3_0_0
 edge_2_9_0_0 edge_2_13_0_0 edge_2_20_0_0 edge_2_21_0_0 edge_2_22_0_0
 edge_2_26_0_0 edge_3_13_0_0 edge_9_1_0_0 edge_9_3_0_0 edge_9_13_0_0
 edge_9_22_0_0 edge_13_3_0_0 edge_13_20_0_0 edge_13_21_0_0 edge_13_22_0_0
 edge_20_13_0_0 edge_20_21_0_0 edge_20_22_0_0 edge_21_1_0_0 edge_21_2_0_0
 edge_21_9_0_0 edge_21_13_0_0 edge_21_20_0_0 edge_22_1_0_0 edge_22_2_0_0
 edge_22_3_0_0 edge_22_9_0_0 edge_22_13_0_0 edge_22_20_0_0 edge_22_21_0_0
 edge_22_26_0_0
End
