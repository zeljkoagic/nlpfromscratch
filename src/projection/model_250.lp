Maximize
  - 3.6552628840168 edge_0_1_0_0 - 2.1307638753016 edge_0_2_0_0
   - 2.567191320992 edge_0_3_0_0 - 1.84586629383 edge_0_7_0_0
   + 0.35837606603 edge_0_8_0_0 - 0.033548112228364 edge_1_2_0_0
   - 0.5820910361857679 edge_1_3_0_0 - 0.5786611353644985 edge_1_6_0_0
   - 0.5933542556581013 edge_1_7_0_0 - 0.6430337786357173 edge_1_8_0_0
   - 0.2252378681106943 edge_1_14_0_0 + 0.2296467809716129 edge_2_1_0_0
   + 0.1779931207457113 edge_2_3_0_0 - 0.375329126344805 edge_2_4_0_0
   + 0.0638693570276058 edge_2_6_0_0 + 0.065911716017791 edge_2_7_0_0
   - 0.0141253664049658 edge_2_8_0_0 + 0.0778948270253305 edge_2_14_0_0
   - 0.3062565453932034 edge_3_1_0_0 + 0.1709901897857619 edge_3_2_0_0
   + 0.118020735594881 edge_3_4_0_0 + 0.3821064913962038 edge_3_6_0_0
   + 1.136049167987343 edge_3_7_0_0 + 0.5999340221526571 edge_3_8_0_0
   + 1.44023351430623 edge_3_14_0_0 - 0.8767750500485446 edge_4_1_0_0
   - 0.5094536268125019 edge_4_2_0_0 + 0.0962566830261257 edge_4_6_0_0
   - 0.4263730281565361 edge_4_7_0_0 + 0.1589762662131385 edge_4_8_0_0
   + 0.7167052090862497 edge_4_14_0_0 - 0.0152301865548164 edge_6_1_0_0
   + 0.3322778101071091 edge_6_2_0_0 + 0.6938760318048866 edge_6_3_0_0
   - 0.4304173754245809 edge_6_4_0_0 + 0.428533387786579 edge_6_7_0_0
   + 0.0890220136501997 edge_6_8_0_0 - 0.1126498844756259 edge_6_14_0_0
   - 0.5001779786242473 edge_7_1_0_0 + 0.00601186644992288 edge_7_2_0_0
   + 0.1429427853567836 edge_7_3_0_0 + 0.09104019898045 edge_7_4_0_0
   + 1.120140303927305 edge_7_6_0_0 + 0.2995729785432008 edge_7_8_0_0
   + 0.474085009357925 edge_7_14_0_0 + 0.942656086652374 edge_8_1_0_0
   + 0.3764536060689723 edge_8_2_0_0 + 1.246857273594468 edge_8_3_0_0
   + 1.724544267903411 edge_8_4_0_0 + 0.5648560363988906 edge_8_6_0_0
   + 0.6945648571686316 edge_8_7_0_0 + 3.560449770046991 edge_8_14_0_0
   - 1.335735399681046 edge_14_4_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0 - flow_1_14_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_14_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_6_3_0_0
   + edge_7_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_14_0_0 + flow_6_3_0_0
   + flow_7_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0
   + edge_8_4_0_0 + edge_14_4_0_0 = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 - flow_4_1_0_0
   - flow_4_2_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   - flow_4_14_0_0 + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_14_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_1_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_14_0_0
   + flow_7_6_0_0 + flow_8_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_14_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_14_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: edge_1_14_0_0 + edge_2_14_0_0 + edge_3_14_0_0
   + edge_4_14_0_0 + edge_6_14_0_0 + edge_7_14_0_0 + edge_8_14_0_0 = 1
 consume_one_flow_unit_14: flow_1_14_0_0 + flow_2_14_0_0 + flow_3_14_0_0
   + flow_4_14_0_0 + flow_6_14_0_0 + flow_7_14_0_0 + flow_8_14_0_0
   - flow_14_4_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_7_0_0
   + flow_0_8_0_0 = 14
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_14_0_0: flow_1_14_0_0 - 1000 edge_1_14_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_14_0_0: flow_2_14_0_0 - 1000 edge_2_14_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_14_0_0: flow_3_14_0_0 - 1000 edge_3_14_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_14_0_0: flow_4_14_0_0 - 1000 edge_4_14_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_14_0_0: flow_6_14_0_0 - 1000 edge_6_14_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_14_0_0: flow_7_14_0_0 - 1000 edge_7_14_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_14_0_0: flow_8_14_0_0 - 1000 edge_8_14_0_0 <= 0
 inactive_no_flow_14_4_0_0: flow_14_4_0_0 - 1000 edge_14_4_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_7_0_0 edge_0_8_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0
 edge_1_14_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_14_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_14_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0
 edge_4_14_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_6_14_0_0 edge_7_1_0_0 edge_7_2_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_14_0_0
 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0 edge_8_6_0_0
 edge_8_7_0_0 edge_8_14_0_0 edge_14_4_0_0
End
