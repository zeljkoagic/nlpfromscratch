Maximize
  - 3.0104086698624 edge_0_1_0_0 + 1.689940387020983 edge_1_3_0_0
   + 0.3087744654138052 edge_1_11_0_0 - 0.0750609538775761 edge_1_13_0_0
   + 0.00845417451924045 edge_1_15_0_0 + 0.2851698100763883 edge_3_1_0_0
   + 0.0642080685405276 edge_3_11_0_0 - 0.1910749277439491 edge_11_1_0_0
   - 0.0407562699172837 edge_11_3_0_0 - 0.032348128418639 edge_11_13_0_0
   - 0.618502125629933 edge_11_14_0_0 - 0.0306351873315653 edge_11_15_0_0
   + 0.0193204562148624 edge_11_16_0_0 + 0.608874535972377 edge_11_18_0_0
   + 0.2132954847351111 edge_11_20_0_0 + 0.00551894426099673 edge_11_25_0_0
   - 0.2051468189941881 edge_13_1_0_0 - 0.1819795253707684 edge_13_3_0_0
   - 0.0422755915094693 edge_13_11_0_0 - 0.003468645548972 edge_13_14_0_0
   + 0.1151180443433462 edge_13_15_0_0
   - 5.9089342758293487e-04 edge_13_20_0_0
   + 0.0670318079473586 edge_13_25_0_0 + 1.65267785612178 edge_13_26_0_0
   - 0.1183527157921887 edge_14_13_0_0 + 0.0775490409637099 edge_14_15_0_0
   - 0.0641700830639334 edge_15_13_0_0 + 0.2715693002690929 edge_15_14_0_0
   + 1.167128216509602 edge_15_26_0_0 - 0.00269363622238618 edge_16_11_0_0
   + 0.0504667333669112 edge_16_18_0_0 + 0.00762310835367234 edge_16_21_0_0
   - 1.337983856493546 edge_20_1_0_0 - 1.392214116167535 edge_20_3_0_0
   + 0.4766061419131873 edge_20_11_0_0 - 0.4903696056518973 edge_20_13_0_0
   - 0.526121499883148 edge_20_14_0_0 + 0.6475197293842448 edge_20_16_0_0
   + 0.0871054688411067 edge_20_18_0_0 + 0.0891644756695713 edge_20_21_0_0
   + 0.3308540053994813 edge_20_23_0_0 - 0.0222584124695617 edge_20_24_0_0
   + 0.5203811829978651 edge_20_25_0_0 + 2.723858166559957 edge_20_26_0_0
   - 0.0334297090398262 edge_23_11_0_0 - 0.004364614852806 edge_23_16_0_0
   + 0.0148473534603947 edge_23_18_0_0 + 0.1391448837415935 edge_23_21_0_0
   + 0.2142412163073387 edge_23_24_0_0 + 0.35975431201427 edge_23_25_0_0
   + 0.6143988004784753 edge_23_26_0_0 - 0.1652399728204271 edge_24_21_0_0
   + 0.0881474079931764 edge_24_23_0_0 - 2.540013489585272 edge_25_1_0_0
   - 0.2847850366435507 edge_25_11_0_0 - 0.2212042485667335 edge_25_16_0_0
   - 0.275129854077704 edge_25_18_0_0 - 0.2427696371620495 edge_25_20_0_0
   + 0.0593185240613634 edge_25_21_0_0 + 0.0849203167943087 edge_25_23_0_0
   + 0.8178201726597027 edge_25_24_0_0 + 1.275437620915531 edge_25_26_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_11_1_0_0 + edge_13_1_0_0
   + edge_20_1_0_0 + edge_25_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_11_0_0
   - flow_1_13_0_0 - flow_1_15_0_0 + flow_3_1_0_0 + flow_11_1_0_0
   + flow_13_1_0_0 + flow_20_1_0_0 + flow_25_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_1_3_0_0 + edge_11_3_0_0 + edge_13_3_0_0 + edge_20_3_0_0
   = 1
 consume_one_flow_unit_3: flow_1_3_0_0 - flow_3_1_0_0 - flow_3_11_0_0
   + flow_11_3_0_0 + flow_13_3_0_0 + flow_20_3_0_0 = 1
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
 one_parent_11: edge_1_11_0_0 + edge_3_11_0_0 + edge_13_11_0_0
   + edge_16_11_0_0 + edge_20_11_0_0 + edge_23_11_0_0 + edge_25_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_3_11_0_0 - flow_11_1_0_0
   - flow_11_3_0_0 - flow_11_13_0_0 - flow_11_14_0_0 - flow_11_15_0_0
   - flow_11_16_0_0 - flow_11_18_0_0 - flow_11_20_0_0 - flow_11_25_0_0
   + flow_13_11_0_0 + flow_16_11_0_0 + flow_20_11_0_0 + flow_23_11_0_0
   + flow_25_11_0_0 = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: edge_1_13_0_0 + edge_11_13_0_0 + edge_14_13_0_0
   + edge_15_13_0_0 + edge_20_13_0_0 = 1
 consume_one_flow_unit_13: flow_1_13_0_0 + flow_11_13_0_0 - flow_13_1_0_0
   - flow_13_3_0_0 - flow_13_11_0_0 - flow_13_14_0_0 - flow_13_15_0_0
   - flow_13_20_0_0 - flow_13_25_0_0 - flow_13_26_0_0 + flow_14_13_0_0
   + flow_15_13_0_0 + flow_20_13_0_0 = 1
 one_parent_14: edge_11_14_0_0 + edge_13_14_0_0 + edge_15_14_0_0
   + edge_20_14_0_0 = 1
 consume_one_flow_unit_14: flow_11_14_0_0 + flow_13_14_0_0 - flow_14_13_0_0
   - flow_14_15_0_0 + flow_15_14_0_0 + flow_20_14_0_0 = 1
 one_parent_15: edge_1_15_0_0 + edge_11_15_0_0 + edge_13_15_0_0
   + edge_14_15_0_0 = 1
 consume_one_flow_unit_15: flow_1_15_0_0 + flow_11_15_0_0 + flow_13_15_0_0
   + flow_14_15_0_0 - flow_15_13_0_0 - flow_15_14_0_0 - flow_15_26_0_0 = 1
 one_parent_16: edge_11_16_0_0 + edge_20_16_0_0 + edge_23_16_0_0
   + edge_25_16_0_0 = 1
 consume_one_flow_unit_16: flow_11_16_0_0 - flow_16_11_0_0 - flow_16_18_0_0
   - flow_16_21_0_0 + flow_20_16_0_0 + flow_23_16_0_0 + flow_25_16_0_0 = 1
 one_parent_17: = 1
 consume_one_flow_unit_17: = 1
 one_parent_18: edge_11_18_0_0 + edge_16_18_0_0 + edge_20_18_0_0
   + edge_23_18_0_0 + edge_25_18_0_0 = 1
 consume_one_flow_unit_18: flow_11_18_0_0 + flow_16_18_0_0 + flow_20_18_0_0
   + flow_23_18_0_0 + flow_25_18_0_0 = 1
 one_parent_19: = 1
 consume_one_flow_unit_19: = 1
 one_parent_20: edge_11_20_0_0 + edge_13_20_0_0 + edge_25_20_0_0 = 1
 consume_one_flow_unit_20: flow_11_20_0_0 + flow_13_20_0_0 - flow_20_1_0_0
   - flow_20_3_0_0 - flow_20_11_0_0 - flow_20_13_0_0 - flow_20_14_0_0
   - flow_20_16_0_0 - flow_20_18_0_0 - flow_20_21_0_0 - flow_20_23_0_0
   - flow_20_24_0_0 - flow_20_25_0_0 - flow_20_26_0_0 + flow_25_20_0_0 = 1
 one_parent_21: edge_16_21_0_0 + edge_20_21_0_0 + edge_23_21_0_0
   + edge_24_21_0_0 + edge_25_21_0_0 = 1
 consume_one_flow_unit_21: flow_16_21_0_0 + flow_20_21_0_0 + flow_23_21_0_0
   + flow_24_21_0_0 + flow_25_21_0_0 = 1
 one_parent_22: = 1
 consume_one_flow_unit_22: = 1
 one_parent_23: edge_20_23_0_0 + edge_24_23_0_0 + edge_25_23_0_0 = 1
 consume_one_flow_unit_23: flow_20_23_0_0 - flow_23_11_0_0 - flow_23_16_0_0
   - flow_23_18_0_0 - flow_23_21_0_0 - flow_23_24_0_0 - flow_23_25_0_0
   - flow_23_26_0_0 + flow_24_23_0_0 + flow_25_23_0_0 = 1
 one_parent_24: edge_20_24_0_0 + edge_23_24_0_0 + edge_25_24_0_0 = 1
 consume_one_flow_unit_24: flow_20_24_0_0 + flow_23_24_0_0 - flow_24_21_0_0
   - flow_24_23_0_0 + flow_25_24_0_0 = 1
 one_parent_25: edge_11_25_0_0 + edge_13_25_0_0 + edge_20_25_0_0
   + edge_23_25_0_0 = 1
 consume_one_flow_unit_25: flow_11_25_0_0 + flow_13_25_0_0 + flow_20_25_0_0
   + flow_23_25_0_0 - flow_25_1_0_0 - flow_25_11_0_0 - flow_25_16_0_0
   - flow_25_18_0_0 - flow_25_20_0_0 - flow_25_21_0_0 - flow_25_23_0_0
   - flow_25_24_0_0 - flow_25_26_0_0 = 1
 one_parent_26: edge_13_26_0_0 + edge_15_26_0_0 + edge_20_26_0_0
   + edge_23_26_0_0 + edge_25_26_0_0 = 1
 consume_one_flow_unit_26: flow_13_26_0_0 + flow_15_26_0_0 + flow_20_26_0_0
   + flow_23_26_0_0 + flow_25_26_0_0 = 1
 root_flow: flow_0_1_0_0 = 26
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_13_0_0: flow_1_13_0_0 - 1000 edge_1_13_0_0 <= 0
 inactive_no_flow_1_15_0_0: flow_1_15_0_0 - 1000 edge_1_15_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_13_0_0: flow_11_13_0_0 - 1000 edge_11_13_0_0 <= 0
 inactive_no_flow_11_14_0_0: flow_11_14_0_0 - 1000 edge_11_14_0_0 <= 0
 inactive_no_flow_11_15_0_0: flow_11_15_0_0 - 1000 edge_11_15_0_0 <= 0
 inactive_no_flow_11_16_0_0: flow_11_16_0_0 - 1000 edge_11_16_0_0 <= 0
 inactive_no_flow_11_18_0_0: flow_11_18_0_0 - 1000 edge_11_18_0_0 <= 0
 inactive_no_flow_11_20_0_0: flow_11_20_0_0 - 1000 edge_11_20_0_0 <= 0
 inactive_no_flow_11_25_0_0: flow_11_25_0_0 - 1000 edge_11_25_0_0 <= 0
 inactive_no_flow_13_1_0_0: flow_13_1_0_0 - 1000 edge_13_1_0_0 <= 0
 inactive_no_flow_13_3_0_0: flow_13_3_0_0 - 1000 edge_13_3_0_0 <= 0
 inactive_no_flow_13_11_0_0: flow_13_11_0_0 - 1000 edge_13_11_0_0 <= 0
 inactive_no_flow_13_14_0_0: flow_13_14_0_0 - 1000 edge_13_14_0_0 <= 0
 inactive_no_flow_13_15_0_0: flow_13_15_0_0 - 1000 edge_13_15_0_0 <= 0
 inactive_no_flow_13_20_0_0: flow_13_20_0_0 - 1000 edge_13_20_0_0 <= 0
 inactive_no_flow_13_25_0_0: flow_13_25_0_0 - 1000 edge_13_25_0_0 <= 0
 inactive_no_flow_13_26_0_0: flow_13_26_0_0 - 1000 edge_13_26_0_0 <= 0
 inactive_no_flow_14_13_0_0: flow_14_13_0_0 - 1000 edge_14_13_0_0 <= 0
 inactive_no_flow_14_15_0_0: flow_14_15_0_0 - 1000 edge_14_15_0_0 <= 0
 inactive_no_flow_15_13_0_0: flow_15_13_0_0 - 1000 edge_15_13_0_0 <= 0
 inactive_no_flow_15_14_0_0: flow_15_14_0_0 - 1000 edge_15_14_0_0 <= 0
 inactive_no_flow_15_26_0_0: flow_15_26_0_0 - 1000 edge_15_26_0_0 <= 0
 inactive_no_flow_16_11_0_0: flow_16_11_0_0 - 1000 edge_16_11_0_0 <= 0
 inactive_no_flow_16_18_0_0: flow_16_18_0_0 - 1000 edge_16_18_0_0 <= 0
 inactive_no_flow_16_21_0_0: flow_16_21_0_0 - 1000 edge_16_21_0_0 <= 0
 inactive_no_flow_20_1_0_0: flow_20_1_0_0 - 1000 edge_20_1_0_0 <= 0
 inactive_no_flow_20_3_0_0: flow_20_3_0_0 - 1000 edge_20_3_0_0 <= 0
 inactive_no_flow_20_11_0_0: flow_20_11_0_0 - 1000 edge_20_11_0_0 <= 0
 inactive_no_flow_20_13_0_0: flow_20_13_0_0 - 1000 edge_20_13_0_0 <= 0
 inactive_no_flow_20_14_0_0: flow_20_14_0_0 - 1000 edge_20_14_0_0 <= 0
 inactive_no_flow_20_16_0_0: flow_20_16_0_0 - 1000 edge_20_16_0_0 <= 0
 inactive_no_flow_20_18_0_0: flow_20_18_0_0 - 1000 edge_20_18_0_0 <= 0
 inactive_no_flow_20_21_0_0: flow_20_21_0_0 - 1000 edge_20_21_0_0 <= 0
 inactive_no_flow_20_23_0_0: flow_20_23_0_0 - 1000 edge_20_23_0_0 <= 0
 inactive_no_flow_20_24_0_0: flow_20_24_0_0 - 1000 edge_20_24_0_0 <= 0
 inactive_no_flow_20_25_0_0: flow_20_25_0_0 - 1000 edge_20_25_0_0 <= 0
 inactive_no_flow_20_26_0_0: flow_20_26_0_0 - 1000 edge_20_26_0_0 <= 0
 inactive_no_flow_23_11_0_0: flow_23_11_0_0 - 1000 edge_23_11_0_0 <= 0
 inactive_no_flow_23_16_0_0: flow_23_16_0_0 - 1000 edge_23_16_0_0 <= 0
 inactive_no_flow_23_18_0_0: flow_23_18_0_0 - 1000 edge_23_18_0_0 <= 0
 inactive_no_flow_23_21_0_0: flow_23_21_0_0 - 1000 edge_23_21_0_0 <= 0
 inactive_no_flow_23_24_0_0: flow_23_24_0_0 - 1000 edge_23_24_0_0 <= 0
 inactive_no_flow_23_25_0_0: flow_23_25_0_0 - 1000 edge_23_25_0_0 <= 0
 inactive_no_flow_23_26_0_0: flow_23_26_0_0 - 1000 edge_23_26_0_0 <= 0
 inactive_no_flow_24_21_0_0: flow_24_21_0_0 - 1000 edge_24_21_0_0 <= 0
 inactive_no_flow_24_23_0_0: flow_24_23_0_0 - 1000 edge_24_23_0_0 <= 0
 inactive_no_flow_25_1_0_0: flow_25_1_0_0 - 1000 edge_25_1_0_0 <= 0
 inactive_no_flow_25_11_0_0: flow_25_11_0_0 - 1000 edge_25_11_0_0 <= 0
 inactive_no_flow_25_16_0_0: flow_25_16_0_0 - 1000 edge_25_16_0_0 <= 0
 inactive_no_flow_25_18_0_0: flow_25_18_0_0 - 1000 edge_25_18_0_0 <= 0
 inactive_no_flow_25_20_0_0: flow_25_20_0_0 - 1000 edge_25_20_0_0 <= 0
 inactive_no_flow_25_21_0_0: flow_25_21_0_0 - 1000 edge_25_21_0_0 <= 0
 inactive_no_flow_25_23_0_0: flow_25_23_0_0 - 1000 edge_25_23_0_0 <= 0
 inactive_no_flow_25_24_0_0: flow_25_24_0_0 - 1000 edge_25_24_0_0 <= 0
 inactive_no_flow_25_26_0_0: flow_25_26_0_0 - 1000 edge_25_26_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_1_3_0_0 edge_1_11_0_0 edge_1_13_0_0 edge_1_15_0_0
 edge_3_1_0_0 edge_3_11_0_0 edge_11_1_0_0 edge_11_3_0_0 edge_11_13_0_0
 edge_11_14_0_0 edge_11_15_0_0 edge_11_16_0_0 edge_11_18_0_0 edge_11_20_0_0
 edge_11_25_0_0 edge_13_1_0_0 edge_13_3_0_0 edge_13_11_0_0 edge_13_14_0_0
 edge_13_15_0_0 edge_13_20_0_0 edge_13_25_0_0 edge_13_26_0_0 edge_14_13_0_0
 edge_14_15_0_0 edge_15_13_0_0 edge_15_14_0_0 edge_15_26_0_0 edge_16_11_0_0
 edge_16_18_0_0 edge_16_21_0_0 edge_20_1_0_0 edge_20_3_0_0 edge_20_11_0_0
 edge_20_13_0_0 edge_20_14_0_0 edge_20_16_0_0 edge_20_18_0_0 edge_20_21_0_0
 edge_20_23_0_0 edge_20_24_0_0 edge_20_25_0_0 edge_20_26_0_0 edge_23_11_0_0
 edge_23_16_0_0 edge_23_18_0_0 edge_23_21_0_0 edge_23_24_0_0 edge_23_25_0_0
 edge_23_26_0_0 edge_24_21_0_0 edge_24_23_0_0 edge_25_1_0_0 edge_25_11_0_0
 edge_25_16_0_0 edge_25_18_0_0 edge_25_20_0_0 edge_25_21_0_0 edge_25_23_0_0
 edge_25_24_0_0 edge_25_26_0_0
End
