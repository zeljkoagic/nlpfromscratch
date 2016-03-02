Maximize
  - 4.9324382184795 edge_0_1_0_0 - 1.739057702676 edge_0_2_0_0
   - 1.0042714129338 edge_0_12_0_0 - 1.0070170140147 edge_0_16_0_0
   - 2.2689862702306 edge_0_24_0_0 - 0.0776571227989379 edge_1_2_0_0
   - 0.3220965724354213 edge_1_12_0_0 + 0.160999370069193 edge_1_13_0_0
   - 0.5814654896632414 edge_1_24_0_0 + 0.1039784986402618 edge_2_1_0_0
   - 0.1189098877904341 edge_2_12_0_0 + 0.1131843355512231 edge_2_13_0_0
   - 0.1035948074018562 edge_2_16_0_0 - 0.0670556613625572 edge_2_22_0_0
   - 0.1091620399473945 edge_2_24_0_0 - 0.3792341540719512 edge_12_1_0_0
   - 0.1126340642305565 edge_12_2_0_0 + 0.0529444874824515 edge_12_13_0_0
   + 0.0464987205224262 edge_12_14_0_0 + 0.077935971549019 edge_12_16_0_0
   + 0.3651361562253896 edge_12_22_0_0 + 0.00651271042761503 edge_12_24_0_0
   + 0.3508527372187281 edge_12_25_0_0 + 0.3395195802176741 edge_12_26_0_0
   - 0.5088579971162428 edge_13_1_0_0 - 0.1597754374862131 edge_13_2_0_0
   - 0.1108166231833874 edge_13_16_0_0 + 0.1686388657771915 edge_13_22_0_0
   - 0.0955540995169975 edge_14_12_0_0 - 0.0902377350436294 edge_14_13_0_0
   - 0.0231991229377312 edge_16_1_0_0 - 0.1092857634661218 edge_16_2_0_0
   + 0.0313037654047446 edge_16_12_0_0 + 0.1428315058215355 edge_16_13_0_0
   - 0.00641302429508364 edge_16_22_0_0 + 0.0807181544695279 edge_16_24_0_0
   - 0.0285179282233407 edge_16_25_0_0 - 0.7320971563573309 edge_22_1_0_0
   - 0.2345896427139628 edge_22_2_0_0 + 0.0659678125985445 edge_22_12_0_0
   + 0.6076698682835844 edge_22_13_0_0 + 0.0118787598511022 edge_22_16_0_0
   - 0.1751243074990302 edge_22_24_0_0 + 0.2003197706832459 edge_22_25_0_0
   + 0.1194164693261077 edge_22_26_0_0 - 0.4950351175761836 edge_24_1_0_0
   - 0.1299510473097348 edge_24_2_0_0 + 0.3097598319619066 edge_24_12_0_0
   + 0.0273277239340078 edge_24_13_0_0 + 0.7734380722864583 edge_24_14_0_0
   + 0.1071835391631528 edge_24_16_0_0 + 0.5279447992141129 edge_24_22_0_0
   + 0.0267733574375424 edge_24_25_0_0 + 0.0822944020076234 edge_24_26_0_0
   - 0.1140601546929773 edge_26_22_0_0 - 0.1124162982408052 edge_26_25_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_12_1_0_0 + edge_13_1_0_0
   + edge_16_1_0_0 + edge_22_1_0_0 + edge_24_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_12_0_0
   - flow_1_13_0_0 - flow_1_24_0_0 + flow_2_1_0_0 + flow_12_1_0_0
   + flow_13_1_0_0 + flow_16_1_0_0 + flow_22_1_0_0 + flow_24_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_12_2_0_0 + edge_13_2_0_0
   + edge_16_2_0_0 + edge_22_2_0_0 + edge_24_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_12_0_0 - flow_2_13_0_0 - flow_2_16_0_0 - flow_2_22_0_0
   - flow_2_24_0_0 + flow_12_2_0_0 + flow_13_2_0_0 + flow_16_2_0_0
   + flow_22_2_0_0 + flow_24_2_0_0 = 1
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
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_2_12_0_0
   + edge_14_12_0_0 + edge_16_12_0_0 + edge_22_12_0_0 + edge_24_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_2_12_0_0
   - flow_12_1_0_0 - flow_12_2_0_0 - flow_12_13_0_0 - flow_12_14_0_0
   - flow_12_16_0_0 - flow_12_22_0_0 - flow_12_24_0_0 - flow_12_25_0_0
   - flow_12_26_0_0 + flow_14_12_0_0 + flow_16_12_0_0 + flow_22_12_0_0
   + flow_24_12_0_0 = 1
 one_parent_13: edge_1_13_0_0 + edge_2_13_0_0 + edge_12_13_0_0
   + edge_14_13_0_0 + edge_16_13_0_0 + edge_22_13_0_0 + edge_24_13_0_0 = 1
 consume_one_flow_unit_13: flow_1_13_0_0 + flow_2_13_0_0 + flow_12_13_0_0
   - flow_13_1_0_0 - flow_13_2_0_0 - flow_13_16_0_0 - flow_13_22_0_0
   + flow_14_13_0_0 + flow_16_13_0_0 + flow_22_13_0_0 + flow_24_13_0_0 = 1
 one_parent_14: edge_12_14_0_0 + edge_24_14_0_0 = 1
 consume_one_flow_unit_14: flow_12_14_0_0 - flow_14_12_0_0 - flow_14_13_0_0
   + flow_24_14_0_0 = 1
 one_parent_15: = 1
 consume_one_flow_unit_15: = 1
 one_parent_16: edge_0_16_0_0 + edge_2_16_0_0 + edge_12_16_0_0
   + edge_13_16_0_0 + edge_22_16_0_0 + edge_24_16_0_0 = 1
 consume_one_flow_unit_16: flow_0_16_0_0 + flow_2_16_0_0 + flow_12_16_0_0
   + flow_13_16_0_0 - flow_16_1_0_0 - flow_16_2_0_0 - flow_16_12_0_0
   - flow_16_13_0_0 - flow_16_22_0_0 - flow_16_24_0_0 - flow_16_25_0_0
   + flow_22_16_0_0 + flow_24_16_0_0 = 1
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
 one_parent_22: edge_2_22_0_0 + edge_12_22_0_0 + edge_13_22_0_0
   + edge_16_22_0_0 + edge_24_22_0_0 + edge_26_22_0_0 = 1
 consume_one_flow_unit_22: flow_2_22_0_0 + flow_12_22_0_0 + flow_13_22_0_0
   + flow_16_22_0_0 - flow_22_1_0_0 - flow_22_2_0_0 - flow_22_12_0_0
   - flow_22_13_0_0 - flow_22_16_0_0 - flow_22_24_0_0 - flow_22_25_0_0
   - flow_22_26_0_0 + flow_24_22_0_0 + flow_26_22_0_0 = 1
 one_parent_23: = 1
 consume_one_flow_unit_23: = 1
 one_parent_24: edge_0_24_0_0 + edge_1_24_0_0 + edge_2_24_0_0
   + edge_12_24_0_0 + edge_16_24_0_0 + edge_22_24_0_0 = 1
 consume_one_flow_unit_24: flow_0_24_0_0 + flow_1_24_0_0 + flow_2_24_0_0
   + flow_12_24_0_0 + flow_16_24_0_0 + flow_22_24_0_0 - flow_24_1_0_0
   - flow_24_2_0_0 - flow_24_12_0_0 - flow_24_13_0_0 - flow_24_14_0_0
   - flow_24_16_0_0 - flow_24_22_0_0 - flow_24_25_0_0 - flow_24_26_0_0 = 1
 one_parent_25: edge_12_25_0_0 + edge_16_25_0_0 + edge_22_25_0_0
   + edge_24_25_0_0 + edge_26_25_0_0 = 1
 consume_one_flow_unit_25: flow_12_25_0_0 + flow_16_25_0_0 + flow_22_25_0_0
   + flow_24_25_0_0 + flow_26_25_0_0 = 1
 one_parent_26: edge_12_26_0_0 + edge_22_26_0_0 + edge_24_26_0_0 = 1
 consume_one_flow_unit_26: flow_12_26_0_0 + flow_22_26_0_0 + flow_24_26_0_0
   - flow_26_22_0_0 - flow_26_25_0_0 = 1
 one_parent_27: = 1
 consume_one_flow_unit_27: = 1
 one_parent_28: = 1
 consume_one_flow_unit_28: = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_12_0_0 + flow_0_16_0_0
   + flow_0_24_0_0 = 28
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_0_16_0_0: flow_0_16_0_0 - 1000 edge_0_16_0_0 <= 0
 inactive_no_flow_0_24_0_0: flow_0_24_0_0 - 1000 edge_0_24_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_1_13_0_0: flow_1_13_0_0 - 1000 edge_1_13_0_0 <= 0
 inactive_no_flow_1_24_0_0: flow_1_24_0_0 - 1000 edge_1_24_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_2_16_0_0: flow_2_16_0_0 - 1000 edge_2_16_0_0 <= 0
 inactive_no_flow_2_22_0_0: flow_2_22_0_0 - 1000 edge_2_22_0_0 <= 0
 inactive_no_flow_2_24_0_0: flow_2_24_0_0 - 1000 edge_2_24_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_13_0_0: flow_12_13_0_0 - 1000 edge_12_13_0_0 <= 0
 inactive_no_flow_12_14_0_0: flow_12_14_0_0 - 1000 edge_12_14_0_0 <= 0
 inactive_no_flow_12_16_0_0: flow_12_16_0_0 - 1000 edge_12_16_0_0 <= 0
 inactive_no_flow_12_22_0_0: flow_12_22_0_0 - 1000 edge_12_22_0_0 <= 0
 inactive_no_flow_12_24_0_0: flow_12_24_0_0 - 1000 edge_12_24_0_0 <= 0
 inactive_no_flow_12_25_0_0: flow_12_25_0_0 - 1000 edge_12_25_0_0 <= 0
 inactive_no_flow_12_26_0_0: flow_12_26_0_0 - 1000 edge_12_26_0_0 <= 0
 inactive_no_flow_13_1_0_0: flow_13_1_0_0 - 1000 edge_13_1_0_0 <= 0
 inactive_no_flow_13_2_0_0: flow_13_2_0_0 - 1000 edge_13_2_0_0 <= 0
 inactive_no_flow_13_16_0_0: flow_13_16_0_0 - 1000 edge_13_16_0_0 <= 0
 inactive_no_flow_13_22_0_0: flow_13_22_0_0 - 1000 edge_13_22_0_0 <= 0
 inactive_no_flow_14_12_0_0: flow_14_12_0_0 - 1000 edge_14_12_0_0 <= 0
 inactive_no_flow_14_13_0_0: flow_14_13_0_0 - 1000 edge_14_13_0_0 <= 0
 inactive_no_flow_16_1_0_0: flow_16_1_0_0 - 1000 edge_16_1_0_0 <= 0
 inactive_no_flow_16_2_0_0: flow_16_2_0_0 - 1000 edge_16_2_0_0 <= 0
 inactive_no_flow_16_12_0_0: flow_16_12_0_0 - 1000 edge_16_12_0_0 <= 0
 inactive_no_flow_16_13_0_0: flow_16_13_0_0 - 1000 edge_16_13_0_0 <= 0
 inactive_no_flow_16_22_0_0: flow_16_22_0_0 - 1000 edge_16_22_0_0 <= 0
 inactive_no_flow_16_24_0_0: flow_16_24_0_0 - 1000 edge_16_24_0_0 <= 0
 inactive_no_flow_16_25_0_0: flow_16_25_0_0 - 1000 edge_16_25_0_0 <= 0
 inactive_no_flow_22_1_0_0: flow_22_1_0_0 - 1000 edge_22_1_0_0 <= 0
 inactive_no_flow_22_2_0_0: flow_22_2_0_0 - 1000 edge_22_2_0_0 <= 0
 inactive_no_flow_22_12_0_0: flow_22_12_0_0 - 1000 edge_22_12_0_0 <= 0
 inactive_no_flow_22_13_0_0: flow_22_13_0_0 - 1000 edge_22_13_0_0 <= 0
 inactive_no_flow_22_16_0_0: flow_22_16_0_0 - 1000 edge_22_16_0_0 <= 0
 inactive_no_flow_22_24_0_0: flow_22_24_0_0 - 1000 edge_22_24_0_0 <= 0
 inactive_no_flow_22_25_0_0: flow_22_25_0_0 - 1000 edge_22_25_0_0 <= 0
 inactive_no_flow_22_26_0_0: flow_22_26_0_0 - 1000 edge_22_26_0_0 <= 0
 inactive_no_flow_24_1_0_0: flow_24_1_0_0 - 1000 edge_24_1_0_0 <= 0
 inactive_no_flow_24_2_0_0: flow_24_2_0_0 - 1000 edge_24_2_0_0 <= 0
 inactive_no_flow_24_12_0_0: flow_24_12_0_0 - 1000 edge_24_12_0_0 <= 0
 inactive_no_flow_24_13_0_0: flow_24_13_0_0 - 1000 edge_24_13_0_0 <= 0
 inactive_no_flow_24_14_0_0: flow_24_14_0_0 - 1000 edge_24_14_0_0 <= 0
 inactive_no_flow_24_16_0_0: flow_24_16_0_0 - 1000 edge_24_16_0_0 <= 0
 inactive_no_flow_24_22_0_0: flow_24_22_0_0 - 1000 edge_24_22_0_0 <= 0
 inactive_no_flow_24_25_0_0: flow_24_25_0_0 - 1000 edge_24_25_0_0 <= 0
 inactive_no_flow_24_26_0_0: flow_24_26_0_0 - 1000 edge_24_26_0_0 <= 0
 inactive_no_flow_26_22_0_0: flow_26_22_0_0 - 1000 edge_26_22_0_0 <= 0
 inactive_no_flow_26_25_0_0: flow_26_25_0_0 - 1000 edge_26_25_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_12_0_0 edge_0_16_0_0 edge_0_24_0_0
 edge_1_2_0_0 edge_1_12_0_0 edge_1_13_0_0 edge_1_24_0_0 edge_2_1_0_0
 edge_2_12_0_0 edge_2_13_0_0 edge_2_16_0_0 edge_2_22_0_0 edge_2_24_0_0
 edge_12_1_0_0 edge_12_2_0_0 edge_12_13_0_0 edge_12_14_0_0 edge_12_16_0_0
 edge_12_22_0_0 edge_12_24_0_0 edge_12_25_0_0 edge_12_26_0_0 edge_13_1_0_0
 edge_13_2_0_0 edge_13_16_0_0 edge_13_22_0_0 edge_14_12_0_0 edge_14_13_0_0
 edge_16_1_0_0 edge_16_2_0_0 edge_16_12_0_0 edge_16_13_0_0 edge_16_22_0_0
 edge_16_24_0_0 edge_16_25_0_0 edge_22_1_0_0 edge_22_2_0_0 edge_22_12_0_0
 edge_22_13_0_0 edge_22_16_0_0 edge_22_24_0_0 edge_22_25_0_0 edge_22_26_0_0
 edge_24_1_0_0 edge_24_2_0_0 edge_24_12_0_0 edge_24_13_0_0 edge_24_14_0_0
 edge_24_16_0_0 edge_24_22_0_0 edge_24_25_0_0 edge_24_26_0_0 edge_26_22_0_0
 edge_26_25_0_0
End
