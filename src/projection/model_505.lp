Maximize
  - 3.1915884081024 edge_0_3_0_0 - 2.7715926293634 edge_0_4_0_0
   - 3.2446677738735 edge_0_8_0_0 - 2.7823016304432 edge_0_10_0_0
   - 2.148729820192 edge_0_11_0_0 - 3.097212458856 edge_0_14_0_0
   - 2.9697021193776 edge_0_15_0_0 + 0.0921070393946702 edge_3_4_0_0
   + 0.1487593205231942 edge_3_8_0_0 - 0.1075856855190598 edge_3_10_0_0
   - 0.1748679443601637 edge_3_11_0_0 - 0.8813063607596922 edge_3_14_0_0
   - 1.102320793723394 edge_3_15_0_0 + 0.1012925499117381 edge_3_16_0_0
   + 0.0102915363954303 edge_4_3_0_0 + 0.3583243456273488 edge_4_8_0_0
   - 0.0238851382280127 edge_4_10_0_0 + 0.7590851859678128 edge_4_16_0_0
   + 0.2519051184496385 edge_8_3_0_0 + 0.4935765792849119 edge_8_4_0_0
   - 0.4606530118907234 edge_8_10_0_0 - 0.4218283329984256 edge_8_11_0_0
   - 0.7314316444532146 edge_8_12_0_0 - 1.473490568224467 edge_8_14_0_0
   - 2.135383752795495 edge_8_15_0_0 + 0.1654346584552623 edge_10_3_0_0
   + 0.4307696920810917 edge_10_4_0_0 + 0.3330757791981936 edge_10_8_0_0
   - 0.2233947345684482 edge_10_12_0_0 - 0.7197581584217196 edge_10_15_0_0
   + 0.8337604199859293 edge_10_16_0_0 + 1.63850782292909 edge_11_3_0_0
   + 1.693753324849066 edge_11_4_0_0 + 2.51504062738129 edge_11_8_0_0
   + 3.227614120240604 edge_11_10_0_0 + 0.0785388846585388 edge_11_12_0_0
   - 1.146859450142991 edge_11_14_0_0 - 0.8540744582378337 edge_11_15_0_0
   + 2.682832848326268 edge_11_16_0_0 - 0.8965216780181675 edge_12_4_0_0
   - 1.12613822891329 edge_12_8_0_0 - 0.657466188653193 edge_12_10_0_0
   - 0.8076472502959795 edge_12_11_0_0 - 0.8178510908115569 edge_12_14_0_0
   - 0.4444017545043069 edge_12_16_0_0 - 0.4813631764944218 edge_14_3_0_0
   - 0.8747674500368969 edge_14_4_0_0 - 1.314593347217887 edge_14_8_0_0
   + 0.1633204679198912 edge_14_10_0_0 - 0.863183 edge_14_11_0_0
   - 0.3563617984199378 edge_14_12_0_0 + 0.2130818036816474 edge_14_16_0_0
   + 0.112101290781979 edge_15_3_0_0 - 0.0562899271835805 edge_15_4_0_0
   - 0.1545009303589274 edge_15_8_0_0 - 0.6948528616072253 edge_15_10_0_0
   + 0.0797441411769182 edge_15_11_0_0 + 0.3906419020065377 edge_15_12_0_0
   + 3.089604696546824 edge_15_14_0_0 + 0.8064893321850566 edge_15_16_0_0
   - 1.438371799096845 edge_16_15_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_0_3_0_0 + edge_4_3_0_0 + edge_8_3_0_0 + edge_10_3_0_0
   + edge_11_3_0_0 + edge_14_3_0_0 + edge_15_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 - flow_3_4_0_0 - flow_3_8_0_0
   - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_14_0_0 - flow_3_15_0_0
   - flow_3_16_0_0 + flow_4_3_0_0 + flow_8_3_0_0 + flow_10_3_0_0
   + flow_11_3_0_0 + flow_14_3_0_0 + flow_15_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_3_4_0_0 + edge_8_4_0_0 + edge_10_4_0_0
   + edge_11_4_0_0 + edge_12_4_0_0 + edge_14_4_0_0 + edge_15_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_3_4_0_0 - flow_4_3_0_0
   - flow_4_8_0_0 - flow_4_10_0_0 - flow_4_16_0_0 + flow_8_4_0_0
   + flow_10_4_0_0 + flow_11_4_0_0 + flow_12_4_0_0 + flow_14_4_0_0
   + flow_15_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_10_8_0_0
   + edge_11_8_0_0 + edge_12_8_0_0 + edge_14_8_0_0 + edge_15_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_10_0_0 - flow_8_11_0_0
   - flow_8_12_0_0 - flow_8_14_0_0 - flow_8_15_0_0 + flow_10_8_0_0
   + flow_11_8_0_0 + flow_12_8_0_0 + flow_14_8_0_0 + flow_15_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_0_10_0_0 + edge_3_10_0_0 + edge_4_10_0_0
   + edge_8_10_0_0 + edge_11_10_0_0 + edge_12_10_0_0 + edge_14_10_0_0
   + edge_15_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_3_10_0_0 + flow_4_10_0_0
   + flow_8_10_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_8_0_0
   - flow_10_12_0_0 - flow_10_15_0_0 - flow_10_16_0_0 + flow_11_10_0_0
   + flow_12_10_0_0 + flow_14_10_0_0 + flow_15_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_3_11_0_0 + edge_8_11_0_0
   + edge_12_11_0_0 + edge_14_11_0_0 + edge_15_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_3_11_0_0 + flow_8_11_0_0
   - flow_11_3_0_0 - flow_11_4_0_0 - flow_11_8_0_0 - flow_11_10_0_0
   - flow_11_12_0_0 - flow_11_14_0_0 - flow_11_15_0_0 - flow_11_16_0_0
   + flow_12_11_0_0 + flow_14_11_0_0 + flow_15_11_0_0 = 1
 one_parent_12: edge_8_12_0_0 + edge_10_12_0_0 + edge_11_12_0_0
   + edge_14_12_0_0 + edge_15_12_0_0 = 1
 consume_one_flow_unit_12: flow_8_12_0_0 + flow_10_12_0_0 + flow_11_12_0_0
   - flow_12_4_0_0 - flow_12_8_0_0 - flow_12_10_0_0 - flow_12_11_0_0
   - flow_12_14_0_0 - flow_12_16_0_0 + flow_14_12_0_0 + flow_15_12_0_0 = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: edge_0_14_0_0 + edge_3_14_0_0 + edge_8_14_0_0
   + edge_11_14_0_0 + edge_12_14_0_0 + edge_15_14_0_0 = 1
 consume_one_flow_unit_14: flow_0_14_0_0 + flow_3_14_0_0 + flow_8_14_0_0
   + flow_11_14_0_0 + flow_12_14_0_0 - flow_14_3_0_0 - flow_14_4_0_0
   - flow_14_8_0_0 - flow_14_10_0_0 - flow_14_11_0_0 - flow_14_12_0_0
   - flow_14_16_0_0 + flow_15_14_0_0 = 1
 one_parent_15: edge_0_15_0_0 + edge_3_15_0_0 + edge_8_15_0_0
   + edge_10_15_0_0 + edge_11_15_0_0 + edge_16_15_0_0 = 1
 consume_one_flow_unit_15: flow_0_15_0_0 + flow_3_15_0_0 + flow_8_15_0_0
   + flow_10_15_0_0 + flow_11_15_0_0 - flow_15_3_0_0 - flow_15_4_0_0
   - flow_15_8_0_0 - flow_15_10_0_0 - flow_15_11_0_0 - flow_15_12_0_0
   - flow_15_14_0_0 - flow_15_16_0_0 + flow_16_15_0_0 = 1
 one_parent_16: edge_3_16_0_0 + edge_4_16_0_0 + edge_10_16_0_0
   + edge_11_16_0_0 + edge_12_16_0_0 + edge_14_16_0_0 + edge_15_16_0_0 = 1
 consume_one_flow_unit_16: flow_3_16_0_0 + flow_4_16_0_0 + flow_10_16_0_0
   + flow_11_16_0_0 + flow_12_16_0_0 + flow_14_16_0_0 + flow_15_16_0_0
   - flow_16_15_0_0 = 1
 root_flow: flow_0_3_0_0 + flow_0_4_0_0 + flow_0_8_0_0 + flow_0_10_0_0
   + flow_0_11_0_0 + flow_0_14_0_0 + flow_0_15_0_0 = 16
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_14_0_0: flow_0_14_0_0 - 1000 edge_0_14_0_0 <= 0
 inactive_no_flow_0_15_0_0: flow_0_15_0_0 - 1000 edge_0_15_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_14_0_0: flow_3_14_0_0 - 1000 edge_3_14_0_0 <= 0
 inactive_no_flow_3_15_0_0: flow_3_15_0_0 - 1000 edge_3_15_0_0 <= 0
 inactive_no_flow_3_16_0_0: flow_3_16_0_0 - 1000 edge_3_16_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_16_0_0: flow_4_16_0_0 - 1000 edge_4_16_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_8_14_0_0: flow_8_14_0_0 - 1000 edge_8_14_0_0 <= 0
 inactive_no_flow_8_15_0_0: flow_8_15_0_0 - 1000 edge_8_15_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_10_15_0_0: flow_10_15_0_0 - 1000 edge_10_15_0_0 <= 0
 inactive_no_flow_10_16_0_0: flow_10_16_0_0 - 1000 edge_10_16_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_11_14_0_0: flow_11_14_0_0 - 1000 edge_11_14_0_0 <= 0
 inactive_no_flow_11_15_0_0: flow_11_15_0_0 - 1000 edge_11_15_0_0 <= 0
 inactive_no_flow_11_16_0_0: flow_11_16_0_0 - 1000 edge_11_16_0_0 <= 0
 inactive_no_flow_12_4_0_0: flow_12_4_0_0 - 1000 edge_12_4_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
 inactive_no_flow_12_14_0_0: flow_12_14_0_0 - 1000 edge_12_14_0_0 <= 0
 inactive_no_flow_12_16_0_0: flow_12_16_0_0 - 1000 edge_12_16_0_0 <= 0
 inactive_no_flow_14_3_0_0: flow_14_3_0_0 - 1000 edge_14_3_0_0 <= 0
 inactive_no_flow_14_4_0_0: flow_14_4_0_0 - 1000 edge_14_4_0_0 <= 0
 inactive_no_flow_14_8_0_0: flow_14_8_0_0 - 1000 edge_14_8_0_0 <= 0
 inactive_no_flow_14_10_0_0: flow_14_10_0_0 - 1000 edge_14_10_0_0 <= 0
 inactive_no_flow_14_11_0_0: flow_14_11_0_0 - 1000 edge_14_11_0_0 <= 0
 inactive_no_flow_14_12_0_0: flow_14_12_0_0 - 1000 edge_14_12_0_0 <= 0
 inactive_no_flow_14_16_0_0: flow_14_16_0_0 - 1000 edge_14_16_0_0 <= 0
 inactive_no_flow_15_3_0_0: flow_15_3_0_0 - 1000 edge_15_3_0_0 <= 0
 inactive_no_flow_15_4_0_0: flow_15_4_0_0 - 1000 edge_15_4_0_0 <= 0
 inactive_no_flow_15_8_0_0: flow_15_8_0_0 - 1000 edge_15_8_0_0 <= 0
 inactive_no_flow_15_10_0_0: flow_15_10_0_0 - 1000 edge_15_10_0_0 <= 0
 inactive_no_flow_15_11_0_0: flow_15_11_0_0 - 1000 edge_15_11_0_0 <= 0
 inactive_no_flow_15_12_0_0: flow_15_12_0_0 - 1000 edge_15_12_0_0 <= 0
 inactive_no_flow_15_14_0_0: flow_15_14_0_0 - 1000 edge_15_14_0_0 <= 0
 inactive_no_flow_15_16_0_0: flow_15_16_0_0 - 1000 edge_15_16_0_0 <= 0
 inactive_no_flow_16_15_0_0: flow_16_15_0_0 - 1000 edge_16_15_0_0 <= 0
Bounds
Binaries
 edge_0_3_0_0 edge_0_4_0_0 edge_0_8_0_0 edge_0_10_0_0 edge_0_11_0_0
 edge_0_14_0_0 edge_0_15_0_0 edge_3_4_0_0 edge_3_8_0_0 edge_3_10_0_0
 edge_3_11_0_0 edge_3_14_0_0 edge_3_15_0_0 edge_3_16_0_0 edge_4_3_0_0
 edge_4_8_0_0 edge_4_10_0_0 edge_4_16_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_10_0_0 edge_8_11_0_0 edge_8_12_0_0 edge_8_14_0_0 edge_8_15_0_0
 edge_10_3_0_0 edge_10_4_0_0 edge_10_8_0_0 edge_10_12_0_0 edge_10_15_0_0
 edge_10_16_0_0 edge_11_3_0_0 edge_11_4_0_0 edge_11_8_0_0 edge_11_10_0_0
 edge_11_12_0_0 edge_11_14_0_0 edge_11_15_0_0 edge_11_16_0_0 edge_12_4_0_0
 edge_12_8_0_0 edge_12_10_0_0 edge_12_11_0_0 edge_12_14_0_0 edge_12_16_0_0
 edge_14_3_0_0 edge_14_4_0_0 edge_14_8_0_0 edge_14_10_0_0 edge_14_11_0_0
 edge_14_12_0_0 edge_14_16_0_0 edge_15_3_0_0 edge_15_4_0_0 edge_15_8_0_0
 edge_15_10_0_0 edge_15_11_0_0 edge_15_12_0_0 edge_15_14_0_0 edge_15_16_0_0
 edge_16_15_0_0
End
