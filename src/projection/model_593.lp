Maximize
  - 4.3368075892477 edge_0_2_0_0 - 1.5141681832448 edge_0_3_0_0
   - 3.847980743831 edge_0_4_0_0 - 2.454500722144 edge_0_5_0_0
   - 1.53340306527 edge_0_6_0_0 - 2.118115653378 edge_0_7_0_0
   - 1.8055591333312 edge_0_10_0_0 - 0.1265380116576952 edge_2_3_0_0
   + 0.0961066458976731 edge_2_4_0_0 - 0.3281310668958983 edge_2_5_0_0
   - 0.4792081749368742 edge_2_6_0_0 - 1.01403760737213 edge_2_7_0_0
   - 1.435325621777771 edge_2_10_0_0 + 0.3517342589154959 edge_2_11_0_0
   + 3.001712755811482 edge_3_2_0_0 + 2.256676897180177 edge_3_4_0_0
   - 0.3770758593653427 edge_3_5_0_0 + 1.115154600335709 edge_3_6_0_0
   - 0.2984719851921266 edge_3_7_0_0 + 0.1040376337863918 edge_3_10_0_0
   + 2.18785190492239 edge_3_11_0_0 - 1.085659310893062 edge_4_3_0_0
   - 0.2904676350099115 edge_4_5_0_0 - 0.0882140229950628 edge_4_6_0_0
   - 0.1267063882461844 edge_4_10_0_0 - 0.2748682941157262 edge_4_11_0_0
   - 0.3257938400079346 edge_5_2_0_0 + 0.0826928022213475 edge_5_3_0_0
   + 0.073820944013693 edge_5_4_0_0 + 0.8165941082570077 edge_5_6_0_0
   + 0.4332971231656995 edge_5_7_0_0 + 0.1911762362284426 edge_5_10_0_0
   + 0.6616927033541676 edge_5_11_0_0 - 1.078732518293626 edge_6_2_0_0
   + 0.7269505841099715 edge_6_3_0_0 - 0.1345505674088474 edge_6_4_0_0
   + 1.722477872396602 edge_6_5_0_0 + 2.031462840578012 edge_6_7_0_0
   + 1.645627216956257 edge_6_10_0_0 + 1.606943183002417 edge_6_11_0_0
   - 0.6855151647527519 edge_7_2_0_0 + 0.0329366187164711 edge_7_3_0_0
   - 0.5487172487703593 edge_7_4_0_0 + 0.2149776729382499 edge_7_5_0_0
   + 0.7873450561394277 edge_7_6_0_0 + 0.8615212763216689 edge_7_10_0_0
   + 0.4089880040997037 edge_7_11_0_0 + 0.4005654925157813 edge_10_2_0_0
   + 0.1406545048946053 edge_10_3_0_0 + 0.5678408839080649 edge_10_4_0_0
   + 1.073458331171354 edge_10_5_0_0 + 1.84656955071739 edge_10_6_0_0
   + 3.296017231343769 edge_10_7_0_0 + 1.499778290251275 edge_10_11_0_0
   - 0.7852760826800734 edge_11_10_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_10_0_0
   - flow_2_11_0_0 + flow_3_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0
   + flow_7_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0
   - flow_3_10_0_0 - flow_3_11_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_10_0_0
   - flow_4_11_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_10_0_0 - flow_5_11_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 + flow_7_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0 - flow_7_3_0_0
   - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_10_0_0
   - flow_7_11_0_0 + flow_10_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_5_0_0
   - flow_10_6_0_0 - flow_10_7_0_0 - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_2_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_5_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_2_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_5_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0 + flow_10_11_0_0
   - flow_11_10_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_10_0_0 = 11
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_10_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_10_0_0
 edge_3_11_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_10_0_0
 edge_4_11_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_10_0_0 edge_5_11_0_0 edge_6_2_0_0 edge_6_3_0_0
 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_10_0_0 edge_6_11_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_10_0_0 edge_7_11_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0
 edge_10_5_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_11_0_0 edge_11_10_0_0
End
