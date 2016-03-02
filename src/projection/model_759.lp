Maximize
  - 2.198984428944 edge_0_1_0_0 - 2.4088696173836 edge_0_2_0_0
   - 2.1694569397906 edge_0_3_0_0 - 3.0441470552597 edge_0_4_0_0
   - 3.7458693628515 edge_0_6_0_0 - 1.5425621056098 edge_0_7_0_0
   - 2.685128334054 edge_0_8_0_0 - 4.384100016643201 edge_0_9_0_0
   - 5.84457 edge_0_10_0_0 + 0.2204686379075919 edge_1_2_0_0
   + 0.5286694255466233 edge_1_3_0_0 + 0.138083281228656 edge_1_4_0_0
   - 0.3308717548039503 edge_1_7_0_0 - 0.4933890840763631 edge_1_9_0_0
   - 0.4595023509597519 edge_1_10_0_0 + 0.1024020803215153 edge_2_1_0_0
   + 0.4211414824406916 edge_2_3_0_0 + 0.094996675660998 edge_2_4_0_0
   + 0.1234111844498049 edge_2_6_0_0 - 0.0561846625345847 edge_2_7_0_0
   - 0.198594689702809 edge_2_8_0_0 - 0.3360471972068257 edge_2_9_0_0
   - 0.14266450798592 edge_2_10_0_0 + 1.399016966474199 edge_3_1_0_0
   + 3.13926117961073 edge_3_2_0_0 + 1.702563824284963 edge_3_4_0_0
   + 2.121088129352791 edge_3_6_0_0 + 0.8711565101341977 edge_3_7_0_0
   - 0.1169095403187218 edge_3_8_0_0 + 0.4926735546075438 edge_3_9_0_0
   + 1.350811705966176 edge_3_10_0_0 - 0.3136126207358601 edge_4_3_0_0
   - 0.353696888116272 edge_4_7_0_0 - 0.6892318322576619 edge_4_9_0_0
   - 0.393968198100943 edge_4_10_0_0 - 0.5193643049262859 edge_6_2_0_0
   - 0.3038571287989069 edge_6_3_0_0 - 0.1990256730223942 edge_6_7_0_0
   - 1.033726022333796 edge_6_9_0_0 - 1.0194547949124 edge_6_10_0_0
   + 0.1167884411232934 edge_7_1_0_0 + 1.090741089058901 edge_7_2_0_0
   + 1.572042829079997 edge_7_3_0_0 - 0.2627025744736555 edge_7_4_0_0
   + 1.577006320740916 edge_7_6_0_0 + 2.020648177155135 edge_7_8_0_0
   + 3.839936946091338 edge_7_9_0_0 + 4.217991340144178 edge_7_10_0_0
   - 0.1101232021811491 edge_8_2_0_0 + 0.00590123746634989 edge_8_3_0_0
   + 0.1803912764029499 edge_8_6_0_0 - 0.1583119625830136 edge_8_7_0_0
   + 0.4954537175995861 edge_8_9_0_0 + 0.0421206503530653 edge_8_10_0_0
   - 0.1525191380001096 edge_9_1_0_0 + 0.3680195178048276 edge_9_2_0_0
   + 0.2724800176284557 edge_9_3_0_0 + 0.2292612887411691 edge_9_4_0_0
   + 0.7490827428879697 edge_9_6_0_0 + 0.7960092258665219 edge_9_7_0_0
   + 2.222023112339398 edge_9_8_0_0 + 0.9614932111398707 edge_9_10_0_0
   - 0.8442437522079053 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_7_1_0_0
   + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_7_0_0 - flow_1_9_0_0 - flow_1_10_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_7_1_0_0 + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_10_0_0 + flow_3_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0
   + flow_4_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0
   + flow_9_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_7_4_0_0 + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_3_0_0 - flow_4_7_0_0 - flow_4_9_0_0
   - flow_4_10_0_0 + flow_7_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_7_6_0_0
   + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_7_0_0 - flow_6_9_0_0
   - flow_6_10_0_0 + flow_7_6_0_0 + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_7_8_0_0
   + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_7_8_0_0 - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_6_0_0
   - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0 + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0
   - flow_9_4_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0
   + flow_0_10_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_7_0_0 edge_1_9_0_0 edge_1_10_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_6_0_0 edge_2_7_0_0
 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0
 edge_3_10_0_0 edge_4_3_0_0 edge_4_7_0_0 edge_4_9_0_0 edge_4_10_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_7_0_0 edge_6_9_0_0 edge_6_10_0_0
 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_6_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_8_10_0_0 edge_9_1_0_0
 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_6_0_0 edge_9_7_0_0
 edge_9_8_0_0 edge_9_10_0_0 edge_10_9_0_0
End
