Maximize
  - 4.8491697090498 edge_0_2_0_0 - 2.16825 edge_0_3_0_0
   - 3.8157763522602 edge_0_4_0_0 - 3.5988701670777 edge_0_5_0_0
   - 3.94474 edge_0_7_0_0 + 0.96435194213888 edge_2_3_0_0
   + 1.463251140656126 edge_2_4_0_0 + 0.7131622842647204 edge_2_5_0_0
   + 1.017418346236496 edge_2_6_0_0 + 0.32635198041888 edge_2_7_0_0
   + 0.0033383497771615 edge_2_8_0_0 + 0.25227698486338 edge_2_9_0_0
   + 1.507728557906781 edge_2_11_0_0 + 0.5730338968538842 edge_2_12_0_0
   + 2.390249856585 edge_3_2_0_0 + 0.8890249138659699 edge_3_4_0_0
   + 3.1480401893262 edge_3_5_0_0 + 2.033701780768 edge_3_6_0_0
   + 1.12531 edge_3_7_0_0 + 0.9337108486475 edge_3_8_0_0
   + 0.665936 edge_3_9_0_0 + 1.4623599416535 edge_3_11_0_0
   + 0.28346796598384 edge_3_12_0_0 - 0.2835669038899448 edge_4_2_0_0
   + 0.46759424113643 edge_4_3_0_0 + 0.9246331990687522 edge_4_5_0_0
   - 0.1301070183909149 edge_4_6_0_0 - 0.37645401609294 edge_4_7_0_0
   - 0.6824843779227174 edge_4_8_0_0 - 1.1679931766136 edge_4_9_0_0
   - 0.3431082057026238 edge_4_11_0_0 - 1.963243825632013 edge_4_12_0_0
   + 0.9189081343404187 edge_5_2_0_0 + 1.0209246587223 edge_5_3_0_0
   + 3.583874412049757 edge_5_4_0_0 + 2.571862402004551 edge_5_6_0_0
   + 2.1181559285175 edge_5_7_0_0 + 1.487226913206974 edge_5_8_0_0
   + 1.1241600738339 edge_5_9_0_0 + 1.07298137583332 edge_5_11_0_0
   + 0.4661802400719544 edge_5_12_0_0 - 0.42609168671808 edge_6_7_0_0
   - 0.42329697460218 edge_7_2_0_0 + 0.167109 edge_7_3_0_0
   + 0.30304647288123 edge_7_4_0_0 + 0.096594510047292 edge_7_5_0_0
   + 1.4478351701456 edge_7_6_0_0 + 2.8489525574125 edge_7_8_0_0
   + 2.75774 edge_7_9_0_0 + 3.167921562021 edge_7_11_0_0
   + 2.7429596708448 edge_7_12_0_0 - 1.361430062536032 edge_8_6_0_0
   - 0.5129127504462501 edge_8_9_0_0 - 1.3989199160648 edge_9_2_0_0
   - 0.695756 edge_9_3_0_0 - 0.75737836333308 edge_9_5_0_0
   - 0.9335095405059201 edge_9_6_0_0 + 0.545273 edge_9_7_0_0
   + 0.71574026475875 edge_9_8_0_0 + 1.9454987285865 edge_9_11_0_0
   + 1.7061697952596 edge_9_12_0_0 - 1.192588180132648 edge_11_6_0_0
   - 0.834040432468047 edge_11_8_0_0 - 0.5731306430512499 edge_11_9_0_0
   - 0.5087832505917526 edge_11_12_0_0 - 2.333499769232794 edge_12_6_0_0
   - 1.091262343273503 edge_12_8_0_0 - 0.36305195643376 edge_12_9_0_0
   - 0.1953748152009193 edge_12_11_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_7_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0
   - flow_2_9_0_0 - flow_2_11_0_0 - flow_2_12_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_7_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0
   - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_11_0_0 - flow_3_12_0_0
   + flow_4_3_0_0 + flow_5_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_11_0_0
   - flow_4_12_0_0 + flow_5_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_7_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0
   - flow_5_11_0_0 - flow_5_12_0_0 + flow_7_5_0_0 + flow_9_5_0_0 = 1
 one_parent_6: edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 + edge_11_6_0_0
   + edge_12_6_0_0 = 1
 consume_one_flow_unit_6: flow_2_6_0_0 + flow_3_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_7_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 + flow_11_6_0_0 + flow_12_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_11_0_0 - flow_7_12_0_0
   + flow_9_7_0_0 = 1
 one_parent_8: edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_5_8_0_0
   + edge_7_8_0_0 + edge_9_8_0_0 + edge_11_8_0_0 + edge_12_8_0_0 = 1
 consume_one_flow_unit_8: flow_2_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_5_8_0_0 + flow_7_8_0_0 - flow_8_6_0_0 - flow_8_9_0_0
   + flow_9_8_0_0 + flow_11_8_0_0 + flow_12_8_0_0 = 1
 one_parent_9: edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0 + edge_5_9_0_0
   + edge_7_9_0_0 + edge_8_9_0_0 + edge_11_9_0_0 + edge_12_9_0_0 = 1
 consume_one_flow_unit_9: flow_2_9_0_0 + flow_3_9_0_0 + flow_4_9_0_0
   + flow_5_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0 - flow_9_2_0_0
   - flow_9_3_0_0 - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_7_0_0
   - flow_9_8_0_0 - flow_9_11_0_0 - flow_9_12_0_0 + flow_11_9_0_0
   + flow_12_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_2_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_5_11_0_0 + edge_7_11_0_0 + edge_9_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_2_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_5_11_0_0 + flow_7_11_0_0 + flow_9_11_0_0 - flow_11_6_0_0
   - flow_11_8_0_0 - flow_11_9_0_0 - flow_11_12_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_2_12_0_0 + edge_3_12_0_0 + edge_4_12_0_0
   + edge_5_12_0_0 + edge_7_12_0_0 + edge_9_12_0_0 + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_2_12_0_0 + flow_3_12_0_0 + flow_4_12_0_0
   + flow_5_12_0_0 + flow_7_12_0_0 + flow_9_12_0_0 + flow_11_12_0_0
   - flow_12_6_0_0 - flow_12_8_0_0 - flow_12_9_0_0 - flow_12_11_0_0 = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_7_0_0 = 13
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_6_0_0: flow_12_6_0_0 - 1000 edge_12_6_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_7_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0
 edge_2_8_0_0 edge_2_9_0_0 edge_2_11_0_0 edge_2_12_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_3_11_0_0 edge_3_12_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0
 edge_4_11_0_0 edge_4_12_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_11_0_0
 edge_5_12_0_0 edge_6_7_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_7_11_0_0
 edge_7_12_0_0 edge_8_6_0_0 edge_8_9_0_0 edge_9_2_0_0 edge_9_3_0_0
 edge_9_5_0_0 edge_9_6_0_0 edge_9_7_0_0 edge_9_8_0_0 edge_9_11_0_0
 edge_9_12_0_0 edge_11_6_0_0 edge_11_8_0_0 edge_11_9_0_0 edge_11_12_0_0
 edge_12_6_0_0 edge_12_8_0_0 edge_12_9_0_0 edge_12_11_0_0
End
