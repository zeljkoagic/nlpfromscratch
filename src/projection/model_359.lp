Maximize
  - 1.160530577792 edge_0_2_0_0 + 0.28864153528638 edge_0_3_0_0
   - 2.193619339488 edge_0_4_0_0 - 0.5962006687415999 edge_0_5_0_0
   - 3.567849571858 edge_0_6_0_0 - 4.30981 edge_0_8_0_0
   - 5.25659 edge_0_9_0_0 - 0.188940379708023 edge_1_8_0_0
   - 0.4042260654557806 edge_2_1_0_0 + 3.415123886919605 edge_2_3_0_0
   - 0.1094119197204788 edge_2_4_0_0 + 0.730960696712187 edge_2_5_0_0
   + 1.949497329185292 edge_2_6_0_0 + 0.3292498207669596 edge_2_7_0_0
   + 0.9376344340828966 edge_2_8_0_0 + 2.392702029974646 edge_2_9_0_0
   + 0.1055439607764696 edge_3_1_0_0 + 2.498786858479081 edge_3_2_0_0
   + 0.9728270301817596 edge_3_4_0_0 + 0.00730331637322835 edge_3_5_0_0
   + 0.0865144429004356 edge_3_6_0_0 + 1.77718617571965 edge_3_8_0_0
   + 0.1787268423133751 edge_3_9_0_0 + 0.00432128693689226 edge_4_2_0_0
   + 0.6962578744228121 edge_4_3_0_0 + 0.2410323579915498 edge_4_5_0_0
   + 0.2415354431421696 edge_4_8_0_0 + 0.4309717500927869 edge_4_9_0_0
   + 0.324969098288797 edge_5_2_0_0 + 0.1352051269981035 edge_5_3_0_0
   + 0.1315084462180851 edge_5_4_0_0 - 0.2624462946496899 edge_5_8_0_0
   + 0.5786365001635305 edge_5_9_0_0 + 1.124928458846069 edge_6_2_0_0
   + 1.09152927959027 edge_6_3_0_0 + 3.450596958500315 edge_6_5_0_0
   + 0.7372642535149789 edge_6_7_0_0 + 2.6281696846196 edge_6_8_0_0
   + 1.5705898115292 edge_6_9_0_0 - 1.536768929921437 edge_7_2_0_0
   - 0.3069814556671709 edge_7_6_0_0 - 0.24732309090145 edge_7_9_0_0
   - 0.2716057033065762 edge_8_1_0_0 + 0.6705903198222872 edge_8_2_0_0
   + 0.71297861499134 edge_8_3_0_0 + 1.284077641452281 edge_8_4_0_0
   + 0.8671948421764 edge_8_5_0_0 + 0.42448094906228 edge_8_6_0_0
   + 3.175833230093 edge_8_7_0_0 - 0.0328520409188502 edge_8_8_0_0
   + 1.659292395893027 edge_8_9_0_0 - 0.9344448304055591 edge_9_8_0_0
Subject To
 one_parent_1: edge_2_1_0_0 + edge_3_1_0_0 + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: - flow_1_8_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_1_0_0 - flow_2_3_0_0
   - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_9_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_1_0_0
   - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_8_0_0 - flow_3_9_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_8_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 + flow_5_4_0_0 + flow_8_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_8_0_0 - flow_5_9_0_0 + flow_6_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_7_6_0_0
   + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_5_0_0 - flow_6_7_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 + flow_7_6_0_0 + flow_8_6_0_0 = 1
 one_parent_7: edge_2_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_2_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0
   - flow_7_6_0_0 - flow_7_9_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_8_8_0_0
   + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0
   + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_8_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_8_0_0 + flow_0_9_0_0 = 9
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_8_0_0: flow_8_8_0_0 - 1000 edge_8_8_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_8_0_0 edge_0_9_0_0 edge_1_8_0_0 edge_2_1_0_0 edge_2_3_0_0
 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0
 edge_2_9_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_5_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_5_2_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_6_2_0_0 edge_6_3_0_0
 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0 edge_7_2_0_0
 edge_7_6_0_0 edge_7_9_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_8_0_0
 edge_8_9_0_0 edge_9_8_0_0
End
