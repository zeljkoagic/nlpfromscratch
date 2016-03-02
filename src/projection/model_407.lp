Maximize
  - 4.3224642244518 edge_0_2_0_0 - 1.9094408863054 edge_0_3_0_0
   - 3.8105738687985 edge_0_4_0_0 - 2.4306206689623 edge_0_5_0_0
   - 2.3940676814769 edge_0_6_0_0 - 3.3411079774512 edge_0_7_0_0
   - 2.8059398316436 edge_0_9_0_0 - 3.4724624167611 edge_0_10_0_0
   + 1.147345368599384 edge_2_3_0_0 + 0.4154793043872455 edge_2_4_0_0
   - 0.6675621596492005 edge_2_5_0_0 - 0.1250542269170038 edge_2_6_0_0
   - 1.039968357204506 edge_2_7_0_0 - 0.48321529841185 edge_2_9_0_0
   + 0.0626662666633217 edge_2_10_0_0 + 3.21707018141163 edge_3_2_0_0
   + 3.018070921716371 edge_3_4_0_0 + 0.1882990515846859 edge_3_5_0_0
   + 0.414926726175435 edge_3_6_0_0 - 0.2921913348928598 edge_3_7_0_0
   + 0.4053968932883569 edge_3_9_0_0 + 2.439649853621 edge_3_10_0_0
   - 0.5880153982412849 edge_4_2_0_0 - 0.4350098194894093 edge_4_3_0_0
   - 0.5808287353980502 edge_4_5_0_0 - 0.4401281075498736 edge_4_6_0_0
   - 0.8169785997002253 edge_4_7_0_0 - 0.4767539706701966 edge_4_9_0_0
   - 0.6514084058053984 edge_4_10_0_0 - 0.4609570221305486 edge_5_2_0_0
   + 0.6528022831907173 edge_5_3_0_0 + 0.2203880009366788 edge_5_4_0_0
   + 0.6058221131959649 edge_5_7_0_0 + 0.4336393540104459 edge_5_8_0_0
   + 0.8955920024557467 edge_5_9_0_0 + 0.7238167299526229 edge_5_10_0_0
   + 0.1889236974003494 edge_6_2_0_0 + 0.2243403218979399 edge_6_3_0_0
   + 0.6296813553770281 edge_6_4_0_0 - 0.1356621386167236 edge_6_7_0_0
   - 0.0053814639023808 edge_6_9_0_0 + 0.54428195428659 edge_6_10_0_0
   - 0.6392896761055699 edge_7_2_0_0 + 0.1441489244394825 edge_7_3_0_0
   + 0.3258798215259998 edge_7_4_0_0 + 0.3883217828438589 edge_7_5_0_0
   + 0.0702636643962217 edge_7_6_0_0 + 1.166358570517129 edge_7_8_0_0
   + 1.562886326406018 edge_7_9_0_0 + 0.5185520275872959 edge_7_10_0_0
   - 1.460956022963821 edge_8_3_0_0 - 0.9892118785968551 edge_8_5_0_0
   - 0.4704110022989039 edge_8_6_0_0 - 0.3465993891735326 edge_8_7_0_0
   - 1.08664327116 edge_8_10_0_0 + 0.5021832134125571 edge_9_2_0_0
   + 0.7838245449521484 edge_9_3_0_0 + 0.4334522925249186 edge_9_4_0_0
   + 1.078104319700466 edge_9_5_0_0 + 1.316285666478855 edge_9_6_0_0
   + 3.34585741138781 edge_9_7_0_0 + 3.855574922379854 edge_9_8_0_0
   + 1.127819120300489 edge_9_10_0_0 - 1.374486965232172 edge_10_3_0_0
   - 1.637617939637846 edge_10_5_0_0 - 0.9544092555602754 edge_10_9_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_9_0_0
   - flow_2_10_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 + edge_9_3_0_0
   + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0
   - flow_3_9_0_0 - flow_3_10_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0 + flow_9_3_0_0
   + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_9_0_0 - flow_4_10_0_0 + flow_5_4_0_0
   + flow_6_4_0_0 + flow_7_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0 - flow_5_10_0_0
   + flow_7_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_7_0_0 - flow_6_9_0_0 - flow_6_10_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 = 1
 one_parent_8: edge_5_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_5_8_0_0 + flow_7_8_0_0 - flow_8_3_0_0
   - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_10_0_0
   + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0 - flow_9_5_0_0
   - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_10_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_3_0_0 - flow_10_5_0_0
   - flow_10_9_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_9_0_0 + flow_0_10_0_0 = 10
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_9_0_0 edge_2_10_0_0
 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0
 edge_3_9_0_0 edge_3_10_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0
 edge_4_6_0_0 edge_4_7_0_0 edge_4_9_0_0 edge_4_10_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_4_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0
 edge_5_10_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_7_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0
 edge_8_3_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_10_0_0
 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_10_3_0_0 edge_10_5_0_0
 edge_10_9_0_0
End
