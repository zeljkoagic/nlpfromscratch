Maximize
  - 3.16723149466 edge_0_3_0_0 - 4.219538771687 edge_0_4_0_0
   - 3.0330076645823 edge_0_7_0_0 - 4.3113753481984 edge_0_8_0_0
   - 3.3811448611135 edge_0_11_0_0 - 4.171723159236 edge_0_13_0_0
   - 3.4312057285464 edge_0_15_0_0 - 4.931556745170401 edge_0_16_0_0
   + 2.66013192629369 edge_3_4_0_0 + 3.11738 edge_3_7_0_0
   + 2.5886002025127 edge_3_8_0_0 + 1.9814339764224 edge_3_11_0_0
   + 2.839347187952 edge_3_13_0_0 + 2.4530595584492 edge_3_15_0_0
   + 2.56903518289566 edge_3_16_0_0 - 0.9853347100872963 edge_4_3_0_0
   - 0.4097491617808025 edge_4_7_0_0 - 0.3013076956417833 edge_4_8_0_0
   - 0.8643333246166207 edge_4_11_0_0 - 0.6498695817707669 edge_4_13_0_0
   - 0.4513968887929568 edge_4_16_0_0 + 0.4173999219658126 edge_7_3_0_0
   + 1.116962806550977 edge_7_4_0_0 + 2.530870589612146 edge_7_8_0_0
   + 1.418019977289777 edge_7_11_0_0 + 2.0292847555345 edge_7_13_0_0
   + 1.4927397313068 edge_7_15_0_0 + 2.318646684331678 edge_7_16_0_0
   - 1.843652977713063 edge_8_3_0_0 - 0.8494693220211802 edge_8_4_0_0
   - 0.93462777394374 edge_8_7_0_0 - 0.4124157727957705 edge_8_11_0_0
   - 0.4467992124016312 edge_8_13_0_0 - 0.8379601732738783 edge_8_15_0_0
   - 0.5503995710630833 edge_8_16_0_0 - 0.5221794125697601 edge_11_3_0_0
   - 0.013295097786905 edge_11_4_0_0 + 0.5396663798086676 edge_11_7_0_0
   + 0.7814046149628845 edge_11_8_0_0 + 1.944679018928507 edge_11_13_0_0
   + 2.174851827094096 edge_11_15_0_0 + 1.901433360562154 edge_11_16_0_0
   - 1.885899593813696 edge_13_3_0_0 - 1.120745454365432 edge_13_4_0_0
   - 1.455539889565022 edge_13_7_0_0 - 0.7503981798027095 edge_13_8_0_0
   - 0.7523143747058065 edge_13_11_0_0 + 0.0404978543906141 edge_13_15_0_0
   + 0.072113710031335 edge_13_16_0_0 - 1.959249647335 edge_15_3_0_0
   - 2.037013643495087 edge_15_7_0_0 - 0.9642799027244863 edge_15_8_0_0
   - 0.8183093650358079 edge_15_11_0_0 - 0.4623029246682086 edge_15_13_0_0
   + 0.6949099896084641 edge_15_16_0_0 - 2.666314243171438 edge_16_3_0_0
   - 1.534992815390473 edge_16_8_0_0 - 2.136760063590927 edge_16_11_0_0
   - 1.051803818437222 edge_16_13_0_0 - 1.626234392599793 edge_16_15_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_0_3_0_0 + edge_4_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0
   + edge_11_3_0_0 + edge_13_3_0_0 + edge_15_3_0_0 + edge_16_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 - flow_3_4_0_0 - flow_3_7_0_0
   - flow_3_8_0_0 - flow_3_11_0_0 - flow_3_13_0_0 - flow_3_15_0_0
   - flow_3_16_0_0 + flow_4_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0
   + flow_11_3_0_0 + flow_13_3_0_0 + flow_15_3_0_0 + flow_16_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_3_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0
   + edge_11_4_0_0 + edge_13_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_3_4_0_0 - flow_4_3_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_11_0_0 - flow_4_13_0_0
   - flow_4_16_0_0 + flow_7_4_0_0 + flow_8_4_0_0 + flow_11_4_0_0
   + flow_13_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0 + edge_8_7_0_0
   + edge_11_7_0_0 + edge_13_7_0_0 + edge_15_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_3_7_0_0 + flow_4_7_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_8_0_0 - flow_7_11_0_0
   - flow_7_13_0_0 - flow_7_15_0_0 - flow_7_16_0_0 + flow_8_7_0_0
   + flow_11_7_0_0 + flow_13_7_0_0 + flow_15_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_7_8_0_0
   + edge_11_8_0_0 + edge_13_8_0_0 + edge_15_8_0_0 + edge_16_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_7_8_0_0 - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_7_0_0
   - flow_8_11_0_0 - flow_8_13_0_0 - flow_8_15_0_0 - flow_8_16_0_0
   + flow_11_8_0_0 + flow_13_8_0_0 + flow_15_8_0_0 + flow_16_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_0_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_7_11_0_0 + edge_8_11_0_0 + edge_13_11_0_0 + edge_15_11_0_0
   + edge_16_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_7_11_0_0 + flow_8_11_0_0 - flow_11_3_0_0 - flow_11_4_0_0
   - flow_11_7_0_0 - flow_11_8_0_0 - flow_11_13_0_0 - flow_11_15_0_0
   - flow_11_16_0_0 + flow_13_11_0_0 + flow_15_11_0_0 + flow_16_11_0_0 = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: edge_0_13_0_0 + edge_3_13_0_0 + edge_4_13_0_0
   + edge_7_13_0_0 + edge_8_13_0_0 + edge_11_13_0_0 + edge_15_13_0_0
   + edge_16_13_0_0 = 1
 consume_one_flow_unit_13: flow_0_13_0_0 + flow_3_13_0_0 + flow_4_13_0_0
   + flow_7_13_0_0 + flow_8_13_0_0 + flow_11_13_0_0 - flow_13_3_0_0
   - flow_13_4_0_0 - flow_13_7_0_0 - flow_13_8_0_0 - flow_13_11_0_0
   - flow_13_15_0_0 - flow_13_16_0_0 + flow_15_13_0_0 + flow_16_13_0_0 = 1
 one_parent_14: = 1
 consume_one_flow_unit_14: = 1
 one_parent_15: edge_0_15_0_0 + edge_3_15_0_0 + edge_7_15_0_0
   + edge_8_15_0_0 + edge_11_15_0_0 + edge_13_15_0_0 + edge_16_15_0_0 = 1
 consume_one_flow_unit_15: flow_0_15_0_0 + flow_3_15_0_0 + flow_7_15_0_0
   + flow_8_15_0_0 + flow_11_15_0_0 + flow_13_15_0_0 - flow_15_3_0_0
   - flow_15_7_0_0 - flow_15_8_0_0 - flow_15_11_0_0 - flow_15_13_0_0
   - flow_15_16_0_0 + flow_16_15_0_0 = 1
 one_parent_16: edge_0_16_0_0 + edge_3_16_0_0 + edge_4_16_0_0
   + edge_7_16_0_0 + edge_8_16_0_0 + edge_11_16_0_0 + edge_13_16_0_0
   + edge_15_16_0_0 = 1
 consume_one_flow_unit_16: flow_0_16_0_0 + flow_3_16_0_0 + flow_4_16_0_0
   + flow_7_16_0_0 + flow_8_16_0_0 + flow_11_16_0_0 + flow_13_16_0_0
   + flow_15_16_0_0 - flow_16_3_0_0 - flow_16_8_0_0 - flow_16_11_0_0
   - flow_16_13_0_0 - flow_16_15_0_0 = 1
 root_flow: flow_0_3_0_0 + flow_0_4_0_0 + flow_0_7_0_0 + flow_0_8_0_0
   + flow_0_11_0_0 + flow_0_13_0_0 + flow_0_15_0_0 + flow_0_16_0_0 = 16
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_13_0_0: flow_0_13_0_0 - 1000 edge_0_13_0_0 <= 0
 inactive_no_flow_0_15_0_0: flow_0_15_0_0 - 1000 edge_0_15_0_0 <= 0
 inactive_no_flow_0_16_0_0: flow_0_16_0_0 - 1000 edge_0_16_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_13_0_0: flow_3_13_0_0 - 1000 edge_3_13_0_0 <= 0
 inactive_no_flow_3_15_0_0: flow_3_15_0_0 - 1000 edge_3_15_0_0 <= 0
 inactive_no_flow_3_16_0_0: flow_3_16_0_0 - 1000 edge_3_16_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_13_0_0: flow_4_13_0_0 - 1000 edge_4_13_0_0 <= 0
 inactive_no_flow_4_16_0_0: flow_4_16_0_0 - 1000 edge_4_16_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_13_0_0: flow_7_13_0_0 - 1000 edge_7_13_0_0 <= 0
 inactive_no_flow_7_15_0_0: flow_7_15_0_0 - 1000 edge_7_15_0_0 <= 0
 inactive_no_flow_7_16_0_0: flow_7_16_0_0 - 1000 edge_7_16_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_13_0_0: flow_8_13_0_0 - 1000 edge_8_13_0_0 <= 0
 inactive_no_flow_8_15_0_0: flow_8_15_0_0 - 1000 edge_8_15_0_0 <= 0
 inactive_no_flow_8_16_0_0: flow_8_16_0_0 - 1000 edge_8_16_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_13_0_0: flow_11_13_0_0 - 1000 edge_11_13_0_0 <= 0
 inactive_no_flow_11_15_0_0: flow_11_15_0_0 - 1000 edge_11_15_0_0 <= 0
 inactive_no_flow_11_16_0_0: flow_11_16_0_0 - 1000 edge_11_16_0_0 <= 0
 inactive_no_flow_13_3_0_0: flow_13_3_0_0 - 1000 edge_13_3_0_0 <= 0
 inactive_no_flow_13_4_0_0: flow_13_4_0_0 - 1000 edge_13_4_0_0 <= 0
 inactive_no_flow_13_7_0_0: flow_13_7_0_0 - 1000 edge_13_7_0_0 <= 0
 inactive_no_flow_13_8_0_0: flow_13_8_0_0 - 1000 edge_13_8_0_0 <= 0
 inactive_no_flow_13_11_0_0: flow_13_11_0_0 - 1000 edge_13_11_0_0 <= 0
 inactive_no_flow_13_15_0_0: flow_13_15_0_0 - 1000 edge_13_15_0_0 <= 0
 inactive_no_flow_13_16_0_0: flow_13_16_0_0 - 1000 edge_13_16_0_0 <= 0
 inactive_no_flow_15_3_0_0: flow_15_3_0_0 - 1000 edge_15_3_0_0 <= 0
 inactive_no_flow_15_7_0_0: flow_15_7_0_0 - 1000 edge_15_7_0_0 <= 0
 inactive_no_flow_15_8_0_0: flow_15_8_0_0 - 1000 edge_15_8_0_0 <= 0
 inactive_no_flow_15_11_0_0: flow_15_11_0_0 - 1000 edge_15_11_0_0 <= 0
 inactive_no_flow_15_13_0_0: flow_15_13_0_0 - 1000 edge_15_13_0_0 <= 0
 inactive_no_flow_15_16_0_0: flow_15_16_0_0 - 1000 edge_15_16_0_0 <= 0
 inactive_no_flow_16_3_0_0: flow_16_3_0_0 - 1000 edge_16_3_0_0 <= 0
 inactive_no_flow_16_8_0_0: flow_16_8_0_0 - 1000 edge_16_8_0_0 <= 0
 inactive_no_flow_16_11_0_0: flow_16_11_0_0 - 1000 edge_16_11_0_0 <= 0
 inactive_no_flow_16_13_0_0: flow_16_13_0_0 - 1000 edge_16_13_0_0 <= 0
 inactive_no_flow_16_15_0_0: flow_16_15_0_0 - 1000 edge_16_15_0_0 <= 0
Bounds
Binaries
 edge_0_3_0_0 edge_0_4_0_0 edge_0_7_0_0 edge_0_8_0_0 edge_0_11_0_0
 edge_0_13_0_0 edge_0_15_0_0 edge_0_16_0_0 edge_3_4_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_11_0_0 edge_3_13_0_0 edge_3_15_0_0 edge_3_16_0_0
 edge_4_3_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_11_0_0 edge_4_13_0_0
 edge_4_16_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_8_0_0 edge_7_11_0_0
 edge_7_13_0_0 edge_7_15_0_0 edge_7_16_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_7_0_0 edge_8_11_0_0 edge_8_13_0_0 edge_8_15_0_0 edge_8_16_0_0
 edge_11_3_0_0 edge_11_4_0_0 edge_11_7_0_0 edge_11_8_0_0 edge_11_13_0_0
 edge_11_15_0_0 edge_11_16_0_0 edge_13_3_0_0 edge_13_4_0_0 edge_13_7_0_0
 edge_13_8_0_0 edge_13_11_0_0 edge_13_15_0_0 edge_13_16_0_0 edge_15_3_0_0
 edge_15_7_0_0 edge_15_8_0_0 edge_15_11_0_0 edge_15_13_0_0 edge_15_16_0_0
 edge_16_3_0_0 edge_16_8_0_0 edge_16_11_0_0 edge_16_13_0_0 edge_16_15_0_0
End
