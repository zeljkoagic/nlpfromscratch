Maximize
  - 1.5532925103054 edge_0_1_0_0 - 2.2880242208061 edge_0_3_0_0
   - 1.432263408372 edge_0_4_0_0 - 3.6520388748028 edge_0_5_0_0
   - 1.617129682565 edge_0_6_0_0 - 1.7035901078336 edge_0_7_0_0
   - 1.1704343708733 edge_0_9_0_0 + 0.5528600922969957 edge_1_3_0_0
   + 0.0170893240514847 edge_1_4_0_0 + 0.1193789302689702 edge_1_5_0_0
   + 1.57064811476149 edge_1_6_0_0 + 2.270883181735801 edge_1_9_0_0
   + 0.65522441880324 edge_3_1_0_0 + 0.1028926598774773 edge_3_2_0_0
   + 0.2728913691544117 edge_3_4_0_0 + 0.2168807618734934 edge_3_5_0_0
   + 0.5371111915450995 edge_3_6_0_0 + 0.0108368598903886 edge_3_7_0_0
   + 0.0714118963738775 edge_3_8_0_0 + 0.7533726114118406 edge_3_9_0_0
   + 0.4823421443794652 edge_4_1_0_0 + 0.538605623852362 edge_4_2_0_0
   + 0.2560960013072602 edge_4_3_0_0 + 0.5682901297962384 edge_4_4_0_0
   + 1.213706649435954 edge_4_5_0_0 + 0.3865524063379876 edge_4_6_0_0
   + 0.1186641063538642 edge_4_7_0_0 - 0.2656803527272387 edge_4_8_0_0
   + 0.6142044618142983 edge_4_9_0_0 - 0.0779504767502134 edge_5_1_0_0
   + 0.5970297601151541 edge_5_3_0_0 + 0.7694503177877043 edge_5_6_0_0
   + 0.7004956482567987 edge_5_9_0_0 + 0.0789887382855819 edge_6_1_0_0
   + 2.222133556208802 edge_6_3_0_0 - 0.2180459159239583 edge_6_4_0_0
   + 2.76832625896447 edge_6_5_0_0 + 0.483249090199296 edge_6_6_0_0
   + 0.1462540259411853 edge_6_7_0_0 + 0.5111624610895775 edge_6_8_0_0
   + 1.846803364874791 edge_6_9_0_0 + 0.5549844733055458 edge_7_1_0_0
   + 0.108907809370588 edge_7_3_0_0 + 0.1388495748069748 edge_7_4_0_0
   - 0.1143063074331126 edge_7_5_0_0 + 0.442696966707362 edge_7_6_0_0
   + 0.4475114416559747 edge_7_7_0_0 + 0.8461404407821056 edge_7_8_0_0
   + 0.7227434121064674 edge_7_9_0_0 - 0.0358413928584629 edge_8_6_0_0
   - 0.0405622622676926 edge_8_7_0_0 - 0.5842248150532422 edge_8_9_0_0
   - 0.0481665371919215 edge_9_1_0_0 + 0.1162965104787555 edge_9_6_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_4_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_9_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0
   + flow_9_1_0_0 = 1
 one_parent_2: edge_3_2_0_0 + edge_4_2_0_0 = 1
 consume_one_flow_unit_2: flow_3_2_0_0 + flow_4_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_3_4_0_0 + edge_4_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_3_4_0_0
   - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0
   - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0
   + flow_6_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_3_0_0 - flow_5_6_0_0
   - flow_5_9_0_0 + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_6_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0
   + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0
   - flow_6_9_0_0 + flow_7_6_0_0 + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0 + edge_6_7_0_0
   + edge_7_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_3_7_0_0 + flow_4_7_0_0
   + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0
   + flow_8_7_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_4_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0
   = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_4_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_6_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_9_0_0 = 9
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_4_0_0: flow_4_4_0_0 - 1000 edge_4_4_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_6_0_0: flow_6_6_0_0 - 1000 edge_6_6_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_7_0_0: flow_7_7_0_0 - 1000 edge_7_7_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_9_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_5_0_0
 edge_1_6_0_0 edge_1_9_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_4_0_0 edge_4_5_0_0
 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_5_1_0_0
 edge_5_3_0_0 edge_5_6_0_0 edge_5_9_0_0 edge_6_1_0_0 edge_6_3_0_0
 edge_6_4_0_0 edge_6_5_0_0 edge_6_6_0_0 edge_6_7_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_7_1_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0
 edge_7_6_0_0 edge_7_7_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_8_6_0_0
 edge_8_7_0_0 edge_8_9_0_0 edge_9_1_0_0 edge_9_6_0_0
End
