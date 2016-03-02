Maximize
  - 3.8218252820466 edge_0_1_0_0 - 3.27976 edge_0_2_0_0
   - 3.2628515522487 edge_0_3_0_0 - 1.9178359917144 edge_0_4_0_0
   - 4.210994932655 edge_0_5_0_0 - 1.409594604795 edge_0_7_0_0
   - 2.455241558874 edge_0_9_0_0 - 4.1422577631796 edge_0_10_0_0
   - 0.55534453417342 edge_1_2_0_0 - 0.7406315619911297 edge_1_3_0_0
   - 0.705299673379602 edge_1_4_0_0 - 0.9558419118853019 edge_1_5_0_0
   - 1.120392714585751 edge_1_7_0_0 - 0.817872908516591 edge_1_10_0_0
   + 0.35512440238026 edge_2_1_0_0 + 0.20434381190412 edge_2_3_0_0
   + 0.4627420328671301 edge_2_4_0_0 - 0.030796059550025 edge_2_5_0_0
   - 0.318357597753 edge_2_7_0_0 - 0.816174006622 edge_2_9_0_0
   + 0.37813779580548 edge_2_10_0_0 - 0.6080417140833663 edge_3_1_0_0
   - 0.76020646770504 edge_3_2_0_0 + 0.1507292783953398 edge_3_4_0_0
   - 0.1275398531890699 edge_3_5_0_0 + 0.1465208710609606 edge_3_7_0_0
   - 0.6005307841610588 edge_3_9_0_0 - 0.5505250201466185 edge_3_10_0_0
   + 1.755334748183109 edge_4_1_0_0 + 2.1289255505363 edge_4_2_0_0
   + 2.982347751772171 edge_4_3_0_0 + 2.630862781820694 edge_4_5_0_0
   + 2.611519680472461 edge_4_7_0_0 + 1.391974729895534 edge_4_8_0_0
   + 1.623995191896955 edge_4_9_0_0 + 2.764932728210921 edge_4_10_0_0
   - 0.8669465609879609 edge_5_1_0_0 - 1.1850753598495 edge_5_2_0_0
   - 0.2350430117389255 edge_5_3_0_0 - 0.1637568640214289 edge_5_4_0_0
   - 0.0374959937895682 edge_5_7_0_0 - 0.7717736938869041 edge_5_9_0_0
   - 0.3648679636002632 edge_5_10_0_0 - 0.6661112923708682 edge_7_1_0_0
   - 0.8673118816005 edge_7_2_0_0 + 0.5475160542801706 edge_7_3_0_0
   - 1.102274666701132 edge_7_4_0_0 - 0.1405549118681909 edge_7_5_0_0
   + 0.6169399455241349 edge_7_8_0_0 + 2.812388682445488 edge_7_9_0_0
   + 1.989911346527292 edge_7_10_0_0 - 1.355434078938096 edge_8_1_0_0
   - 1.7085148282536 edge_8_2_0_0 - 1.033468662743581 edge_8_4_0_0
   - 0.6875888590639817 edge_8_5_0_0 - 0.3321695968507867 edge_8_7_0_0
   - 0.2264278670304858 edge_8_10_0_0 - 0.5611266350829727 edge_9_1_0_0
   - 0.1114953508942 edge_9_2_0_0 + 1.165920089318723 edge_9_3_0_0
   - 0.2402830132274528 edge_9_4_0_0 + 0.0492380269727826 edge_9_5_0_0
   + 0.3170273266258732 edge_9_7_0_0 + 3.151105835321702 edge_9_8_0_0
   + 0.8769593628696883 edge_9_10_0_0 - 2.178777910767493 edge_10_1_0_0
   - 2.8234084753586 edge_10_2_0_0 - 2.354113808667057 edge_10_4_0_0
   - 1.720976412371936 edge_10_5_0_0 - 1.710213764464068 edge_10_7_0_0
   - 1.122179428975581 edge_10_8_0_0 - 1.529314255957856 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0
   + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_7_0_0 - flow_1_10_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0
   + flow_7_1_0_0 + flow_8_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0
   + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_7_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 + flow_9_2_0_0
   + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_7_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_7_0_0 - flow_3_9_0_0 - flow_3_10_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0
   + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0
   - flow_4_10_0_0 + flow_5_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_9_4_0_0 + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0
   + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_7_0_0 - flow_5_9_0_0
   - flow_5_10_0_0 + flow_7_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0
   + flow_10_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_4_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0
   = 1
 consume_one_flow_unit_8: flow_4_8_0_0 + flow_7_8_0_0 - flow_8_1_0_0
   - flow_8_2_0_0 - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_7_0_0
   - flow_8_10_0_0 + flow_9_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_7_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_7_9_0_0 - flow_9_1_0_0
   - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0 - flow_9_5_0_0
   - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_7_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_7_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_2_0_0
   - flow_10_4_0_0 - flow_10_5_0_0 - flow_10_7_0_0 - flow_10_8_0_0
   - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_7_0_0 + flow_0_9_0_0 + flow_0_10_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_7_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_1_7_0_0 edge_1_10_0_0 edge_2_1_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_7_0_0 edge_2_9_0_0
 edge_2_10_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_7_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0
 edge_4_10_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_7_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_7_1_0_0 edge_7_2_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_8_0_0 edge_7_9_0_0
 edge_7_10_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_4_0_0 edge_8_5_0_0
 edge_8_7_0_0 edge_8_10_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0
 edge_9_4_0_0 edge_9_5_0_0 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0
 edge_10_1_0_0 edge_10_2_0_0 edge_10_4_0_0 edge_10_5_0_0 edge_10_7_0_0
 edge_10_8_0_0 edge_10_9_0_0
End
