Maximize
  - 5.0377545185535 edge_0_1_0_0 - 3.5015624753022 edge_0_2_0_0
   - 1.9094408863054 edge_0_3_0_0 - 3.8470936228955 edge_0_4_0_0
   - 2.3681998454112 edge_0_5_0_0 - 3.1027740294468 edge_0_6_0_0
   - 4.87322 edge_0_7_0_0 + 0.014484149297588 edge_1_2_0_0
   - 0.43813857127079 edge_1_3_0_0 - 0.1220491468333654 edge_1_4_0_0
   - 1.052713372877438 edge_1_5_0_0 - 0.083774875739031 edge_1_7_0_0
   - 0.0515601093520643 edge_2_1_0_0 + 0.8517849488929 edge_2_3_0_0
   + 0.6250338838703448 edge_2_4_0_0 + 0.067166329236034 edge_2_5_0_0
   - 0.3292916743075368 edge_2_6_0_0 + 0.8132056199550866 edge_2_7_0_0
   + 0.8072830582494499 edge_3_1_0_0 + 3.159131829046343 edge_3_2_0_0
   + 2.465059133916568 edge_3_4_0_0 + 0.025980167147652 edge_3_5_0_0
   + 1.595803909043369 edge_3_6_0_0 + 2.880489264870631 edge_3_7_0_0
   - 1.261816981125428 edge_4_1_0_0 - 0.9136634769510481 edge_4_2_0_0
   - 0.57255262952472 edge_4_3_0_0 - 0.6833886534316262 edge_4_5_0_0
   - 0.6158125656230742 edge_4_6_0_0 - 0.90820716491362 edge_4_7_0_0
   + 0.2757369732087145 edge_5_1_0_0 + 0.5330592552041228 edge_5_2_0_0
   - 0.09752590921203 edge_5_3_0_0 + 0.7031146104627501 edge_5_4_0_0
   + 0.6844649691605877 edge_5_6_0_0 + 1.0430022121479 edge_5_7_0_0
   - 0.1670242494462904 edge_6_2_0_0 - 0.8280485495054711 edge_6_3_0_0
   + 0.4538272917891773 edge_6_4_0_0 + 2.147467847391976 edge_6_5_0_0
   + 0.9013574496670506 edge_6_7_0_0 - 0.170299078992 edge_7_5_0_0
   - 1.776291912165364 edge_7_6_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_7_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_6_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 + flow_5_4_0_0
   + flow_6_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_6_5_0_0 + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_7_0_0 = 7
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_7_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0
 edge_1_5_0_0 edge_1_7_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0
 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_7_5_0_0 edge_7_6_0_0
End
