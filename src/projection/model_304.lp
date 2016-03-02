Maximize
  - 1.1557196110632 edge_0_1_0_0 - 3.7750123978956 edge_0_2_0_0
   - 3.151533740628 edge_0_4_0_0 - 3.2282669177265 edge_0_5_0_0
   - 2.9617870427348 edge_0_6_0_0 - 3.7545712634655 edge_0_7_0_0
   - 5.42405 edge_0_8_0_0 + 2.947549746572185 edge_1_2_0_0
   + 1.532726216370372 edge_1_4_0_0 + 2.55886950674941 edge_1_5_0_0
   + 1.705025411382515 edge_1_6_0_0 + 1.599360726427667 edge_1_7_0_0
   + 3.544912333380495 edge_1_8_0_0 + 0.5288682979942743 edge_2_1_0_0
   + 0.6039860050508813 edge_2_4_0_0 + 0.1583064201481629 edge_2_5_0_0
   - 0.1257597141390936 edge_2_6_0_0 - 0.1486465951351477 edge_2_7_0_0
   - 0.2881457589792 edge_2_8_0_0 - 0.614658429012084 edge_4_1_0_0
   + 1.910108277368984 edge_4_2_0_0 + 0.4879361638530337 edge_4_5_0_0
   - 0.2890305016800177 edge_4_6_0_0 - 0.6137727685695349 edge_4_7_0_0
   + 0.2215602552241537 edge_4_8_0_0 + 0.7428560213426018 edge_5_1_0_0
   + 0.9809667213633525 edge_5_2_0_0 + 2.75517286114269 edge_5_4_0_0
   + 0.3717810666687669 edge_5_6_0_0 + 0.2185633609884453 edge_5_7_0_0
   + 1.040712121752726 edge_5_8_0_0 - 0.1758342261324674 edge_6_1_0_0
   + 0.74511345431869 edge_6_2_0_0 - 0.0346273799191964 edge_6_4_0_0
   + 0.3568342257185238 edge_6_5_0_0 + 0.792922165773517 edge_6_7_0_0
   + 0.5325469985795399 edge_6_8_0_0 + 0.5797649838776818 edge_7_1_0_0
   + 1.3948007648049 edge_7_2_0_0 + 0.0609879057277437 edge_7_4_0_0
   - 0.2729779612131219 edge_7_5_0_0 + 3.0473961234465 edge_7_6_0_0
   + 1.8315239360145 edge_7_8_0_0 - 1.1399564593125 edge_8_1_0_0
   - 1.3117726643536 edge_8_2_0_0 - 1.8635737756428 edge_8_5_0_0
   - 0.8819894359273199 edge_8_6_0_0 - 0.46437732061665 edge_8_7_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_4_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_4_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0
   + flow_2_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0
   + flow_7_2_0_0 + flow_8_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 + flow_5_4_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0 + flow_6_5_0_0
   + flow_7_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0
   + flow_7_6_0_0 + flow_8_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_4_8_0_0
   + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_5_0_0 - flow_8_6_0_0
   - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_1_2_0_0 edge_1_4_0_0 edge_1_5_0_0
 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_2_1_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0
 edge_5_1_0_0 edge_5_2_0_0 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0
 edge_5_8_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_8_1_0_0 edge_8_2_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0
End
