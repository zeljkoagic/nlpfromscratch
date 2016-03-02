Maximize
  - 4.490900591458 edge_0_2_0_0 - 3.1672596199288 edge_0_3_0_0
   - 3.3795918102566 edge_0_4_0_0 - 4.4913021400674 edge_0_6_0_0
   - 3.0330094540582 edge_0_7_0_0 - 3.2012138340699 edge_0_8_0_0
   - 3.2247563890924 edge_0_11_0_0 - 3.134928893976 edge_0_12_0_0
   + 0.8814244402236401 edge_2_3_0_0 + 0.3715898210021628 edge_2_4_0_0
   + 0.5958890810798124 edge_2_6_0_0 - 0.14836352475039 edge_2_7_0_0
   + 0.1093349465219047 edge_2_8_0_0 + 0.5421197040724062 edge_2_10_0_0
   - 0.4032216873387076 edge_2_11_0_0 + 0.3654240156478935 edge_2_12_0_0
   + 3.5481530934 edge_3_2_0_0 + 2.660106232222314 edge_3_4_0_0
   + 0.7970294482275999 edge_3_6_0_0 + 3.091426941747922 edge_3_7_0_0
   + 2.565777563049656 edge_3_8_0_0 + 0.39796371954088 edge_3_10_0_0
   + 1.7578598945284 edge_3_11_0_0 + 1.898351435749 edge_3_12_0_0
   - 0.9140624046279148 edge_4_2_0_0 - 0.81937205877783 edge_4_3_0_0
   - 0.2892172992036133 edge_4_6_0_0 + 0.012733032006708 edge_4_7_0_0
   - 0.2174962499571446 edge_4_8_0_0 - 0.4010957477382396 edge_4_11_0_0
   - 0.3597439067483167 edge_4_12_0_0 - 0.9563085707812 edge_6_3_0_0
   - 0.1011482469540976 edge_6_4_0_0 + 0.9379553057819 edge_6_7_0_0
   + 0.243434861347137 edge_6_8_0_0 + 0.3456260981355145 edge_6_10_0_0
   - 0.103276932381941 edge_6_11_0_0 - 0.1646515707126578 edge_6_12_0_0
   + 1.3488487992904 edge_7_2_0_0 + 0.417403874778809 edge_7_3_0_0
   + 1.116942758328467 edge_7_4_0_0 + 3.1068207320772 edge_7_6_0_0
   + 2.552374532596501 edge_7_8_0_0 + 0.6813012983776799 edge_7_10_0_0
   + 2.2616198643028 edge_7_11_0_0 + 1.5269316767648 edge_7_12_0_0
   - 1.572541518246403 edge_8_3_0_0 - 0.5552955219370326 edge_8_4_0_0
   - 0.9345610415609488 edge_8_6_0_0 - 0.767675053205735 edge_8_7_0_0
   - 0.2884740884491079 edge_8_10_0_0 + 0.0127356508241076 edge_8_11_0_0
   - 0.2737532694341152 edge_8_12_0_0 - 1.4960506737952 edge_10_3_0_0
   - 1.004634423758683 edge_10_4_0_0 - 0.9536546892704 edge_10_7_0_0
   - 0.1008883231350776 edge_10_8_0_0 + 0.935352076379093 edge_10_11_0_0
   + 0.1000700976538974 edge_10_12_0_0 + 0.4339351591764132 edge_11_2_0_0
   - 0.3660535724655959 edge_11_3_0_0 + 0.3444902127475481 edge_11_4_0_0
   + 1.030895781331773 edge_11_6_0_0 + 0.9062974620562675 edge_11_7_0_0
   + 1.30082105825632 edge_11_8_0_0 + 3.250589321536629 edge_11_10_0_0
   + 1.540850033809378 edge_11_12_0_0 - 2.341974322053498 edge_12_3_0_0
   - 1.171417353033278 edge_12_4_0_0 - 1.923172466906087 edge_12_7_0_0
   - 0.8956688515892793 edge_12_8_0_0 - 0.6761913343224889 edge_12_11_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_7_2_0_0
   + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_10_0_0
   - flow_2_11_0_0 - flow_2_12_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_7_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_6_3_0_0
   + edge_7_3_0_0 + edge_8_3_0_0 + edge_10_3_0_0 + edge_11_3_0_0
   + edge_12_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_12_0_0 + flow_4_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0 + flow_10_3_0_0
   + flow_11_3_0_0 + flow_12_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0
   + edge_7_4_0_0 + edge_8_4_0_0 + edge_10_4_0_0 + edge_11_4_0_0
   + edge_12_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_6_0_0 - flow_4_7_0_0
   - flow_4_8_0_0 - flow_4_11_0_0 - flow_4_12_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 + flow_8_4_0_0 + flow_10_4_0_0 + flow_11_4_0_0
   + flow_12_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_7_0_0
   - flow_6_8_0_0 - flow_6_10_0_0 - flow_6_11_0_0 - flow_6_12_0_0
   + flow_7_6_0_0 + flow_8_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 + edge_10_7_0_0 + edge_11_7_0_0
   + edge_12_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0 - flow_7_3_0_0
   - flow_7_4_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_10_0_0
   - flow_7_11_0_0 - flow_7_12_0_0 + flow_8_7_0_0 + flow_10_7_0_0
   + flow_11_7_0_0 + flow_12_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0
   + edge_12_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_10_0_0
   - flow_8_11_0_0 - flow_8_12_0_0 + flow_10_8_0_0 + flow_11_8_0_0
   + flow_12_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_2_10_0_0 + edge_3_10_0_0 + edge_6_10_0_0
   + edge_7_10_0_0 + edge_8_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_3_10_0_0 + flow_6_10_0_0
   + flow_7_10_0_0 + flow_8_10_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_7_0_0 - flow_10_8_0_0 - flow_10_11_0_0 - flow_10_12_0_0
   + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_2_11_0_0 + edge_3_11_0_0
   + edge_4_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0 + edge_8_11_0_0
   + edge_10_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_2_11_0_0 + flow_3_11_0_0
   + flow_4_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0 + flow_8_11_0_0
   + flow_10_11_0_0 - flow_11_2_0_0 - flow_11_3_0_0 - flow_11_4_0_0
   - flow_11_6_0_0 - flow_11_7_0_0 - flow_11_8_0_0 - flow_11_10_0_0
   - flow_11_12_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_2_12_0_0 + edge_3_12_0_0
   + edge_4_12_0_0 + edge_6_12_0_0 + edge_7_12_0_0 + edge_8_12_0_0
   + edge_10_12_0_0 + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_2_12_0_0 + flow_3_12_0_0
   + flow_4_12_0_0 + flow_6_12_0_0 + flow_7_12_0_0 + flow_8_12_0_0
   + flow_10_12_0_0 + flow_11_12_0_0 - flow_12_3_0_0 - flow_12_4_0_0
   - flow_12_7_0_0 - flow_12_8_0_0 - flow_12_11_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_11_0_0 + flow_0_12_0_0 = 12
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_4_0_0: flow_12_4_0_0 - 1000 edge_12_4_0_0 <= 0
 inactive_no_flow_12_7_0_0: flow_12_7_0_0 - 1000 edge_12_7_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_0_11_0_0 edge_0_12_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_10_0_0 edge_2_11_0_0
 edge_2_12_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_6_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_3_12_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_11_0_0
 edge_4_12_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_7_0_0 edge_6_8_0_0
 edge_6_10_0_0 edge_6_11_0_0 edge_6_12_0_0 edge_7_2_0_0 edge_7_3_0_0
 edge_7_4_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_10_0_0 edge_7_11_0_0
 edge_7_12_0_0 edge_8_3_0_0 edge_8_4_0_0 edge_8_6_0_0 edge_8_7_0_0
 edge_8_10_0_0 edge_8_11_0_0 edge_8_12_0_0 edge_10_3_0_0 edge_10_4_0_0
 edge_10_7_0_0 edge_10_8_0_0 edge_10_11_0_0 edge_10_12_0_0 edge_11_2_0_0
 edge_11_3_0_0 edge_11_4_0_0 edge_11_6_0_0 edge_11_7_0_0 edge_11_8_0_0
 edge_11_10_0_0 edge_11_12_0_0 edge_12_3_0_0 edge_12_4_0_0 edge_12_7_0_0
 edge_12_8_0_0 edge_12_11_0_0
End
