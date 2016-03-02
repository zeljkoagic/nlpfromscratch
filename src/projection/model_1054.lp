Maximize
  - 2.3880723335688 edge_0_1_0_0 - 4.720582766807 edge_0_2_0_0
   - 2.5068747583268 edge_0_4_0_0 - 3.278107002324 edge_0_5_0_0
   - 1.98413333314 edge_0_6_0_0 - 4.1930325540229 edge_0_7_0_0
   - 4.207854859992699 edge_0_8_0_0 - 3.8594933131462 edge_0_9_0_0
   + 2.303668066264193 edge_1_2_0_0 + 1.127311505240793 edge_1_4_0_0
   + 0.9107680321586582 edge_1_5_0_0 + 2.266548937879593 edge_1_6_0_0
   + 0.5211743289869231 edge_1_7_0_0 + 0.2576686087213838 edge_1_8_0_0
   + 0.3103531842148645 edge_1_9_0_0 + 2.835574223217307 edge_1_10_0_0
   - 0.3613627878738937 edge_2_1_0_0 - 0.2701399212523328 edge_2_4_0_0
   - 0.065620102025249 edge_2_5_0_0 - 0.4123421187256076 edge_2_6_0_0
   - 0.7379024042980806 edge_2_7_0_0 - 0.711992812489132 edge_2_8_0_0
   - 0.4314283605573517 edge_2_10_0_0 + 0.2473299940770941 edge_4_1_0_0
   + 0.3337941915777382 edge_4_2_0_0 + 0.1854259752132336 edge_4_5_0_0
   + 0.7936956856935115 edge_4_6_0_0 - 0.1461815076871676 edge_4_7_0_0
   - 0.1607837217823895 edge_4_8_0_0 + 0.664697813159052 edge_4_9_0_0
   + 0.6213692605221647 edge_4_10_0_0 - 0.0960268393996253 edge_5_1_0_0
   + 0.0848996246381255 edge_5_2_0_0 + 0.6382655458314338 edge_5_4_0_0
   + 0.5381341304293962 edge_5_6_0_0 + 1.142774588585561 edge_5_7_0_0
   + 0.3246889578417245 edge_5_8_0_0 + 0.2427460930097676 edge_5_9_0_0
   + 0.5203666854618001 edge_5_10_0_0 + 2.350490151233238 edge_6_1_0_0
   + 2.592568630597161 edge_6_2_0_0 + 2.450701166266481 edge_6_4_0_0
   + 3.293383230957815 edge_6_5_0_0 + 2.346164559343355 edge_6_7_0_0
   + 2.234130836549331 edge_6_8_0_0 + 2.821320393801918 edge_6_9_0_0
   + 3.214234732880778 edge_6_10_0_0 - 1.239044096485045 edge_7_1_0_0
   - 0.2466463592002714 edge_7_2_0_0 - 0.292103131807502 edge_7_4_0_0
   + 0.0242114988478896 edge_7_5_0_0 + 0.7352549327202751 edge_7_6_0_0
   + 0.2933881479824639 edge_7_8_0_0 + 0.2636421078995826 edge_7_9_0_0
   + 0.3001251295699603 edge_7_10_0_0 - 1.205215659998161 edge_8_1_0_0
   - 0.1641238986862046 edge_8_2_0_0 - 0.581026566133422 edge_8_4_0_0
   - 0.7872878605476167 edge_8_5_0_0 + 0.3679833045141769 edge_8_6_0_0
   - 0.1458237275346791 edge_8_7_0_0 + 1.089503510770522 edge_8_9_0_0
   + 0.37516415595813 edge_8_10_0_0 + 0.5801906756668707 edge_9_1_0_0
   + 0.5030872480397197 edge_9_2_0_0 + 0.206512752051014 edge_9_4_0_0
   + 0.7836654291322643 edge_9_5_0_0 + 0.6571684983225103 edge_9_6_0_0
   + 0.5414586730164478 edge_9_7_0_0 + 3.523276718692741 edge_9_8_0_0
   + 1.2639494456441 edge_9_10_0_0 - 1.170508844955511 edge_10_6_0_0
   - 1.0147965606471 edge_10_8_0_0 - 0.55628194233105 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_4_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_4_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0
   - flow_1_9_0_0 - flow_1_10_0_0 + flow_2_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0
   + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_10_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_10_0_0
   + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0
   - flow_5_10_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0
   + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0
   - flow_6_9_0_0 - flow_6_10_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_4_8_0_0
   + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0
   + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_4_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   + flow_9_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_4_9_0_0 + edge_5_9_0_0
   + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_4_9_0_0
   + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_4_0_0 - flow_9_5_0_0
   - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_10_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_1_10_0_0 + edge_2_10_0_0 + edge_4_10_0_0
   + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0
   + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_1_10_0_0 + flow_2_10_0_0 + flow_4_10_0_0
   + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0
   + flow_9_10_0_0 - flow_10_6_0_0 - flow_10_8_0_0 - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_1_2_0_0 edge_1_4_0_0
 edge_1_5_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_1_9_0_0
 edge_1_10_0_0 edge_2_1_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_10_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0
 edge_4_10_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0
 edge_8_1_0_0 edge_8_2_0_0 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0
 edge_8_7_0_0 edge_8_9_0_0 edge_8_10_0_0 edge_9_1_0_0 edge_9_2_0_0
 edge_9_4_0_0 edge_9_5_0_0 edge_9_6_0_0 edge_9_7_0_0 edge_9_8_0_0
 edge_9_10_0_0 edge_10_6_0_0 edge_10_8_0_0 edge_10_9_0_0
End
