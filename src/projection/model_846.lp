Maximize
  - 1.8426119266504 edge_0_1_0_0 - 5.085164383435 edge_0_2_0_0
   - 2.21193819635 edge_0_4_0_0 - 2.5885520338132 edge_0_8_0_0
   + 0.9591787028292162 edge_1_2_0_0 + 0.1462309903941492 edge_1_3_0_0
   + 0.913431494265581 edge_1_4_0_0 + 0.3208955759899512 edge_1_8_0_0
   - 0.00154649791892421 edge_1_11_0_0 - 0.6651347039710681 edge_1_13_0_0
   + 1.7639771384445 edge_1_14_0_0 + 0.0406052351439133 edge_2_1_0_0
   + 0.8677555586115971 edge_2_3_0_0 + 0.6428248752253509 edge_2_4_0_0
   - 0.3128766857260011 edge_2_8_0_0 - 0.1557732268569206 edge_2_11_0_0
   - 1.321390597364416 edge_2_13_0_0 + 1.2366696037636 edge_2_14_0_0
   - 0.5034096843622625 edge_3_1_0_0 - 0.3374958727732701 edge_3_2_0_0
   + 0.1910867729888648 edge_3_4_0_0 - 0.383875650374039 edge_3_8_0_0
   + 0.0631944809312211 edge_3_11_0_0 + 0.6247869021901925 edge_4_1_0_0
   + 2.893973033710587 edge_4_2_0_0 + 4.289635439765636 edge_4_3_0_0
   + 0.70959238718064 edge_4_5_0_0 + 1.33457412742707 edge_4_8_0_0
   + 1.22409519004588 edge_4_11_0_0 + 0.9658104812159729 edge_4_13_0_0
   + 5.0269769310664 edge_4_14_0_0 + 0.2818922492624457 edge_8_1_0_0
   + 1.383983435610462 edge_8_2_0_0 + 1.897193143995402 edge_8_3_0_0
   + 0.3550320303173882 edge_8_4_0_0 + 1.16991899154299 edge_8_5_0_0
   + 1.146875667602931 edge_8_11_0_0 + 1.220252386505647 edge_8_13_0_0
   + 3.3200928853456 edge_8_14_0_0 - 0.6312390802927818 edge_11_1_0_0
   - 0.0819226273926806 edge_11_13_0_0 - 0.9339034813316885 edge_13_1_0_0
   - 1.520632497083273 edge_13_2_0_0 - 1.706722448646067 edge_13_4_0_0
   - 0.5606033846656451 edge_13_8_0_0 - 0.151760293719494 edge_13_11_0_0
   + 3.03361383459515 edge_13_12_0_0 + 0.63851103106076 edge_13_14_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_8_1_0_0 + edge_11_1_0_0 + edge_13_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_8_0_0 - flow_1_11_0_0 - flow_1_13_0_0
   - flow_1_14_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_8_1_0_0 + flow_11_1_0_0 + flow_13_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_8_2_0_0 + edge_13_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_8_0_0 - flow_2_11_0_0
   - flow_2_13_0_0 - flow_2_14_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_8_2_0_0 + flow_13_2_0_0 = 1
 one_parent_3: edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_8_3_0_0
   = 1
 consume_one_flow_unit_3: flow_1_3_0_0 + flow_2_3_0_0 - flow_3_1_0_0
   - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_8_0_0 - flow_3_11_0_0
   + flow_4_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_8_4_0_0 + edge_13_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_8_0_0 - flow_4_11_0_0 - flow_4_13_0_0
   - flow_4_14_0_0 + flow_8_4_0_0 + flow_13_4_0_0 = 1
 one_parent_5: edge_4_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_4_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_13_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0
   - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_11_0_0
   - flow_8_13_0_0 - flow_8_14_0_0 + flow_13_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_1_11_0_0 + edge_2_11_0_0 + edge_3_11_0_0
   + edge_4_11_0_0 + edge_8_11_0_0 + edge_13_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_2_11_0_0 + flow_3_11_0_0
   + flow_4_11_0_0 + flow_8_11_0_0 - flow_11_1_0_0 - flow_11_13_0_0
   + flow_13_11_0_0 = 1
 one_parent_12: edge_13_12_0_0 = 1
 consume_one_flow_unit_12: flow_13_12_0_0 = 1
 one_parent_13: edge_1_13_0_0 + edge_2_13_0_0 + edge_4_13_0_0
   + edge_8_13_0_0 + edge_11_13_0_0 = 1
 consume_one_flow_unit_13: flow_1_13_0_0 + flow_2_13_0_0 + flow_4_13_0_0
   + flow_8_13_0_0 + flow_11_13_0_0 - flow_13_1_0_0 - flow_13_2_0_0
   - flow_13_4_0_0 - flow_13_8_0_0 - flow_13_11_0_0 - flow_13_12_0_0
   - flow_13_14_0_0 = 1
 one_parent_14: edge_1_14_0_0 + edge_2_14_0_0 + edge_4_14_0_0
   + edge_8_14_0_0 + edge_13_14_0_0 = 1
 consume_one_flow_unit_14: flow_1_14_0_0 + flow_2_14_0_0 + flow_4_14_0_0
   + flow_8_14_0_0 + flow_13_14_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_4_0_0 + flow_0_8_0_0 = 14
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_13_0_0: flow_1_13_0_0 - 1000 edge_1_13_0_0 <= 0
 inactive_no_flow_1_14_0_0: flow_1_14_0_0 - 1000 edge_1_14_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_2_14_0_0: flow_2_14_0_0 - 1000 edge_2_14_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_13_0_0: flow_4_13_0_0 - 1000 edge_4_13_0_0 <= 0
 inactive_no_flow_4_14_0_0: flow_4_14_0_0 - 1000 edge_4_14_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_13_0_0: flow_8_13_0_0 - 1000 edge_8_13_0_0 <= 0
 inactive_no_flow_8_14_0_0: flow_8_14_0_0 - 1000 edge_8_14_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_13_0_0: flow_11_13_0_0 - 1000 edge_11_13_0_0 <= 0
 inactive_no_flow_13_1_0_0: flow_13_1_0_0 - 1000 edge_13_1_0_0 <= 0
 inactive_no_flow_13_2_0_0: flow_13_2_0_0 - 1000 edge_13_2_0_0 <= 0
 inactive_no_flow_13_4_0_0: flow_13_4_0_0 - 1000 edge_13_4_0_0 <= 0
 inactive_no_flow_13_8_0_0: flow_13_8_0_0 - 1000 edge_13_8_0_0 <= 0
 inactive_no_flow_13_11_0_0: flow_13_11_0_0 - 1000 edge_13_11_0_0 <= 0
 inactive_no_flow_13_12_0_0: flow_13_12_0_0 - 1000 edge_13_12_0_0 <= 0
 inactive_no_flow_13_14_0_0: flow_13_14_0_0 - 1000 edge_13_14_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_4_0_0 edge_0_8_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_8_0_0 edge_1_11_0_0 edge_1_13_0_0
 edge_1_14_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_8_0_0
 edge_2_11_0_0 edge_2_13_0_0 edge_2_14_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_8_0_0 edge_3_11_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_4_8_0_0 edge_4_11_0_0 edge_4_13_0_0
 edge_4_14_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_11_0_0 edge_8_13_0_0 edge_8_14_0_0 edge_11_1_0_0
 edge_11_13_0_0 edge_13_1_0_0 edge_13_2_0_0 edge_13_4_0_0 edge_13_8_0_0
 edge_13_11_0_0 edge_13_12_0_0 edge_13_14_0_0
End
