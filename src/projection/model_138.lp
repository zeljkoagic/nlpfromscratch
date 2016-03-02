Maximize
  - 5.3334705978787 edge_0_1_0_0 - 4.99964 edge_0_2_0_0
   - 1.9752282420453 edge_0_3_0_0 - 2.9953269962622 edge_0_4_0_0
   - 3.520546991392 edge_0_7_0_0 - 0.083746164702022 edge_1_2_0_0
   - 0.0960852533603334 edge_1_3_0_0 - 0.685907770899585 edge_1_4_0_0
   - 1.239798198884927 edge_1_7_0_0 - 1.173992772641349 edge_1_10_0_0
   + 0.13080909757457 edge_2_1_0_0 + 0.5177785391766899 edge_2_3_0_0
   + 0.1928357935797 edge_2_4_0_0 - 0.7825791926917001 edge_2_7_0_0
   - 0.9299185461077599 edge_2_10_0_0 + 0.9799865363474763 edge_3_1_0_0
   + 1.692098494031 edge_3_2_0_0 + 2.936147208342059 edge_3_4_0_0
   + 0.1033293156424472 edge_3_5_0_0 + 1.847211652441166 edge_3_7_0_0
   - 0.4721720761265683 edge_3_8_0_0 + 0.7347876975448071 edge_3_10_0_0
   + 0.1807501935372317 edge_4_1_0_0 + 1.107247137498 edge_4_2_0_0
   + 1.814641749757205 edge_4_3_0_0 + 0.1136360311281728 edge_4_5_0_0
   + 0.2523094800676337 edge_4_6_0_0 + 1.868631560608281 edge_4_7_0_0
   - 0.7415732627871686 edge_4_8_0_0 + 0.9089336506204229 edge_4_10_0_0
   - 1.330866452449413 edge_5_1_0_0 + 0.2319169196742757 edge_5_7_0_0
   - 0.4938661534844327 edge_5_10_0_0 - 1.75545767949691 edge_6_1_0_0
   - 0.443310418870601 edge_6_5_0_0 - 0.0514476926187062 edge_6_10_0_0
   - 0.6996368137345038 edge_7_1_0_0 - 0.3886824022056 edge_7_2_0_0
   + 0.00891261649427326 edge_7_3_0_0 + 0.1599096050581859 edge_7_4_0_0
   + 2.51344195827706 edge_7_5_0_0 + 3.274269362548266 edge_7_6_0_0
   + 0.2615433831748702 edge_7_8_0_0 + 2.481915156991888 edge_7_10_0_0
   - 1.169202407804323 edge_10_1_0_0 - 1.2729819784924 edge_10_2_0_0
   + 0.4621949373511606 edge_10_5_0_0 + 0.4471182930991819 edge_10_6_0_0
   + 0.1869595425532861 edge_10_7_0_0 + 3.21812277817969 edge_10_8_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_7_0_0 - flow_1_10_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_7_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_7_0_0 - flow_2_10_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 + flow_7_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_10_0_0 + flow_4_3_0_0
   + flow_7_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   - flow_4_10_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_3_5_0_0 + edge_4_5_0_0 + edge_6_5_0_0 + edge_7_5_0_0
   + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0
   - flow_5_7_0_0 - flow_5_10_0_0 + flow_6_5_0_0 + flow_7_5_0_0
   + flow_10_5_0_0 = 1
 one_parent_6: edge_4_6_0_0 + edge_7_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_4_6_0_0 - flow_6_1_0_0 - flow_6_5_0_0
   - flow_6_10_0_0 + flow_7_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0
   - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_10_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_4_8_0_0 + edge_7_8_0_0 + edge_10_8_0_0
   = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_4_8_0_0 + flow_7_8_0_0
   + flow_10_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_1_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 = 1
 consume_one_flow_unit_10: flow_1_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_5_0_0 - flow_10_6_0_0
   - flow_10_7_0_0 - flow_10_8_0_0 = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_7_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_7_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_7_0_0 edge_1_10_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_7_0_0 edge_2_10_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_10_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_10_0_0
 edge_5_1_0_0 edge_5_7_0_0 edge_5_10_0_0 edge_6_1_0_0 edge_6_5_0_0
 edge_6_10_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_10_0_0 edge_10_1_0_0
 edge_10_2_0_0 edge_10_5_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_8_0_0
End
