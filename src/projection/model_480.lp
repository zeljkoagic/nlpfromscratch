Maximize
  - 0.5814977427522 edge_0_1_0_0 - 2.0156320544988 edge_0_4_0_0
   - 2.0729077735044 edge_0_6_0_0 - 1.5779040455439 edge_0_9_0_0
   - 0.5119880148099 edge_0_10_0_0 + 0.6843409395141621 edge_1_4_0_0
   + 0.0619912227324758 edge_1_6_0_0 - 0.1001555926203462 edge_1_9_0_0
   + 0.3852276007146265 edge_1_10_0_0 + 0.6007487668043305 edge_1_11_0_0
   - 0.1395103568586436 edge_4_1_0_0 - 0.153462633891562 edge_4_6_0_0
   - 0.6615183605783006 edge_4_10_0_0 - 0.4972953213025152 edge_4_11_0_0
   - 0.605638238173531 edge_6_1_0_0 - 0.3729785871444265 edge_6_4_0_0
   - 0.8151631172031161 edge_6_10_0_0 - 0.7878347624149533 edge_6_11_0_0
   - 0.0791665319670158 edge_9_1_0_0 - 0.7342896883873056 edge_9_4_0_0
   - 0.2644129216493781 edge_9_6_0_0 + 0.2607315091732749 edge_9_10_0_0
   - 0.1240415308067146 edge_9_11_0_0 + 0.3722642733353515 edge_10_1_0_0
   + 0.2655365592762315 edge_10_4_0_0 + 0.3813339128820973 edge_10_6_0_0
   + 1.443226238585533 edge_10_9_0_0 + 1.552807470935085 edge_10_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_4_1_0_0 + edge_6_1_0_0 + edge_9_1_0_0
   + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_4_0_0 - flow_1_6_0_0
   - flow_1_9_0_0 - flow_1_10_0_0 - flow_1_11_0_0 + flow_4_1_0_0
   + flow_6_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_6_4_0_0 + edge_9_4_0_0
   + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 - flow_4_1_0_0
   - flow_4_6_0_0 - flow_4_10_0_0 - flow_4_11_0_0 + flow_6_4_0_0
   + flow_9_4_0_0 + flow_10_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_4_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_4_6_0_0
   - flow_6_1_0_0 - flow_6_4_0_0 - flow_6_10_0_0 - flow_6_11_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 - flow_9_1_0_0
   - flow_9_4_0_0 - flow_9_6_0_0 - flow_9_10_0_0 - flow_9_11_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_4_10_0_0
   + edge_6_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_4_10_0_0
   + flow_6_10_0_0 + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_4_0_0
   - flow_10_6_0_0 - flow_10_9_0_0 - flow_10_11_0_0 = 1
 one_parent_11: edge_1_11_0_0 + edge_4_11_0_0 + edge_6_11_0_0
   + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_4_11_0_0 + flow_6_11_0_0
   + flow_9_11_0_0 + flow_10_11_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_4_0_0 + flow_0_6_0_0 + flow_0_9_0_0
   + flow_0_10_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_0_9_0_0 edge_0_10_0_0
 edge_1_4_0_0 edge_1_6_0_0 edge_1_9_0_0 edge_1_10_0_0 edge_1_11_0_0
 edge_4_1_0_0 edge_4_6_0_0 edge_4_10_0_0 edge_4_11_0_0 edge_6_1_0_0
 edge_6_4_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_9_1_0_0 edge_9_4_0_0
 edge_9_6_0_0 edge_9_10_0_0 edge_9_11_0_0 edge_10_1_0_0 edge_10_4_0_0
 edge_10_6_0_0 edge_10_9_0_0 edge_10_11_0_0
End
