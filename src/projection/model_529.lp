Maximize
  - 4.55224 edge_0_2_0_0 - 1.794963306686 edge_0_3_0_0
   - 4.0890987592326 edge_0_6_0_0 - 2.914843877945 edge_0_8_0_0
   + 0.9280643689211 edge_2_3_0_0 - 0.11983965360965 edge_2_4_0_0
   - 0.096721623655831 edge_2_6_0_0 + 0.2410848680335 edge_2_8_0_0
   - 0.10268743769895 edge_2_9_0_0 + 0.879194 edge_2_11_0_0
   + 2.228659273276 edge_3_2_0_0 + 0.670365350927756 edge_3_4_0_0
   + 1.418561838991405 edge_3_6_0_0 + 2.436430696999789 edge_3_8_0_0
   + 0.5559314623237174 edge_3_9_0_0 + 3.131460877143 edge_3_11_0_0
   - 1.552448930357 edge_4_2_0_0 - 1.583720274686468 edge_4_3_0_0
   + 0.1994504679665667 edge_4_5_0_0 - 0.7181553804545922 edge_4_6_0_0
   + 0.0352795559632607 edge_4_8_0_0 - 0.5883990190638281 edge_4_9_0_0
   - 0.7172147332467 edge_4_11_0_0 - 0.8880446423328001 edge_6_2_0_0
   - 0.0979964113734004 edge_6_3_0_0 + 0.2530296329424288 edge_6_4_0_0
   + 0.4738462824485234 edge_6_5_0_0 + 0.2612482105529707 edge_6_8_0_0
   + 2.784890059945309 edge_6_9_0_0 + 0.66515955365113 edge_6_11_0_0
   + 0.4649643138181 edge_8_2_0_0 + 0.5440008184052378 edge_8_3_0_0
   + 3.180912279421324 edge_8_4_0_0 + 3.062352267126795 edge_8_5_0_0
   + 3.041404119334439 edge_8_6_0_0 + 0.8279615500124173 edge_8_9_0_0
   + 1.757294582486 edge_8_11_0_0 - 0.1994256544098 edge_9_11_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_6_2_0_0
   + edge_8_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_6_0_0 - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_11_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 + flow_6_2_0_0 + flow_8_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_6_3_0_0
   + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_6_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   - flow_3_11_0_0 + flow_4_3_0_0 + flow_6_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0 + edge_8_4_0_0
   = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 - flow_4_2_0_0
   - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 - flow_4_11_0_0 + flow_6_4_0_0 + flow_8_4_0_0 = 1
 one_parent_5: edge_4_5_0_0 + edge_6_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_4_5_0_0 + flow_6_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_5_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_11_0_0
   + flow_8_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_6_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_6_8_0_0 - flow_8_2_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_9_0_0
   - flow_8_11_0_0 = 1
 one_parent_9: edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0 + edge_6_9_0_0
   + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_2_9_0_0 + flow_3_9_0_0 + flow_4_9_0_0
   + flow_6_9_0_0 + flow_8_9_0_0 - flow_9_11_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_2_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_6_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0 = 1
 consume_one_flow_unit_11: flow_2_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_6_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_6_0_0 + flow_0_8_0_0 = 11
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_6_0_0 edge_0_8_0_0 edge_2_3_0_0
 edge_2_4_0_0 edge_2_6_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_11_0_0
 edge_3_2_0_0 edge_3_4_0_0 edge_3_6_0_0 edge_3_8_0_0 edge_3_9_0_0
 edge_3_11_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0
 edge_4_8_0_0 edge_4_9_0_0 edge_4_11_0_0 edge_6_2_0_0 edge_6_3_0_0
 edge_6_4_0_0 edge_6_5_0_0 edge_6_8_0_0 edge_6_9_0_0 edge_6_11_0_0
 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0
 edge_8_9_0_0 edge_8_11_0_0 edge_9_11_0_0
End
