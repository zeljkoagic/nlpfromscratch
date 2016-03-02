Maximize
  - 3.3223333588855 edge_0_1_0_0 - 5.727542588217601 edge_0_3_0_0
   - 1.648055296503 edge_0_5_0_0 - 2.523928378074 edge_0_6_0_0
   - 3.0507477701295 edge_0_7_0_0 - 2.8938956539107 edge_0_8_0_0
   - 1.66125 edge_0_9_0_0 - 4.704046895327 edge_0_10_0_0
   - 0.9547818293739151 edge_1_3_0_0 - 0.1771058017841279 edge_1_5_0_0
   - 0.3056272913540072 edge_1_6_0_0 + 0.086123293000874 edge_1_7_0_0
   + 0.1706321818097478 edge_1_8_0_0 - 0.7063623185282411 edge_1_9_0_0
   + 1.205816589794 edge_1_10_0_0 - 0.0700632766454788 edge_2_6_0_0
   - 0.7012133175188845 edge_3_1_0_0 + 2.801133960926672 edge_3_2_0_0
   + 0.6457119881582586 edge_3_4_0_0 + 1.481317252031564 edge_3_5_0_0
   + 1.241093633164886 edge_3_6_0_0 + 0.3661994518194822 edge_3_7_0_0
   + 0.2358052016010079 edge_3_8_0_0 + 3.062229772121193 edge_3_9_0_0
   + 1.654194971232 edge_3_10_0_0 + 0.0879194161106775 edge_5_1_0_0
   + 0.3837624357333346 edge_5_6_0_0 + 0.1016790121695499 edge_5_8_0_0
   - 0.07074493927746 edge_5_9_0_0 - 0.0389731994969214 edge_5_10_0_0
   + 1.288225200723371 edge_6_1_0_0 + 0.6456296879068122 edge_6_3_0_0
   + 3.473129279960735 edge_6_5_0_0 - 0.0786982592815443 edge_6_7_0_0
   + 1.195905862315108 edge_6_8_0_0 + 0.7842465308175001 edge_6_9_0_0
   + 1.121328554191666 edge_6_10_0_0 - 0.1083737230062874 edge_7_1_0_0
   - 0.7714100363098606 edge_7_3_0_0 - 0.1906738874595423 edge_7_6_0_0
   + 0.302798567213718 edge_7_8_0_0 - 0.5898796379442556 edge_7_9_0_0
   - 0.1217632279592 edge_7_10_0_0 + 0.6860175265563991 edge_8_1_0_0
   + 2.148247884387878 edge_8_3_0_0 + 0.1672265626957785 edge_8_4_0_0
   + 0.6083110387486192 edge_8_5_0_0 + 2.747098487399262 edge_8_6_0_0
   + 2.215371391912358 edge_8_7_0_0 + 1.9916588329701 edge_8_9_0_0
   + 3.8451344385126 edge_8_10_0_0 - 0.23473554958101 edge_9_1_0_0
   + 0.5824670787734239 edge_9_2_0_0 + 0.3450698474631748 edge_9_3_0_0
   + 2.699633203216396 edge_9_4_0_0 + 0.7789958884504999 edge_9_5_0_0
   + 1.191586743831 edge_9_6_0_0 + 1.7090660929641 edge_9_8_0_0
   + 2.2521785135612 edge_9_10_0_0 - 2.036289218234929 edge_10_1_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0 + edge_6_1_0_0
   + edge_7_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_5_0_0
   - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0 - flow_1_9_0_0
   - flow_1_10_0_0 + flow_3_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 + flow_8_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_3_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: - flow_2_6_0_0 + flow_3_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0
   + edge_8_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: edge_3_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_3_4_0_0 + flow_8_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   - flow_5_1_0_0 - flow_5_6_0_0 - flow_5_8_0_0 - flow_5_9_0_0
   - flow_5_10_0_0 + flow_6_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_3_0_0
   - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0
   - flow_6_10_0_0 + flow_7_6_0_0 + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_3_7_0_0 + edge_6_7_0_0
   + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_3_7_0_0
   + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_3_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_1_0_0
   - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_6_0_0
   - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0 + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0
   + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_3_9_0_0
   + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0
   - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_8_0_0 - flow_9_10_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_3_10_0_0
   + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0
   + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_3_10_0_0
   + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0
   + flow_9_10_0_0 - flow_10_1_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_10_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_1_3_0_0 edge_1_5_0_0
 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_1_9_0_0 edge_1_10_0_0
 edge_2_6_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0
 edge_5_1_0_0 edge_5_6_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0
 edge_6_1_0_0 edge_6_3_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_7_1_0_0 edge_7_3_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0 edge_8_1_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0
 edge_9_5_0_0 edge_9_6_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_10_1_0_0
End
