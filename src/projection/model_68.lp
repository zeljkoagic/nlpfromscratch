Maximize
  - 1.7120362911135 edge_0_3_0_0 - 4.7517246765594 edge_0_10_0_0
   - 1.422705256854 edge_0_13_0_0 - 1.8154577714427 edge_0_15_0_0
   - 0.2119404672333413 edge_1_3_0_0 + 0.044103762225111 edge_2_3_0_0
   - 0.0537175272243491 edge_2_4_0_0 + 0.194701057126372 edge_2_5_0_0
   + 0.8438070765789906 edge_2_6_0_0 - 0.6887822168908661 edge_2_10_0_0
   - 0.2464759727304228 edge_2_13_0_0 - 0.2613541999419436 edge_2_17_0_0
   + 0.7047744670360162 edge_2_20_0_0 - 0.00574292978127425 edge_3_1_0_0
   + 0.5480216802205361 edge_3_2_0_0 + 0.534143231043722 edge_3_4_0_0
   + 0.8206338403971143 edge_3_5_0_0 + 0.3332464773679804 edge_3_6_0_0
   + 0.0575968628348136 edge_3_10_0_0 - 0.2089559735417333 edge_3_11_0_0
   + 0.3399755626489525 edge_3_13_0_0 - 0.0218207064984805 edge_3_14_0_0
   + 0.3096025886695862 edge_3_15_0_0 + 0.1151958810755403 edge_3_17_0_0
   + 1.710142667750212 edge_3_20_0_0 + 0.0360774384861841 edge_4_1_0_0
   - 0.0590337846150511 edge_4_2_0_0 - 0.0678520334660408 edge_4_5_0_0
   - 0.1017372812499864 edge_4_6_0_0 - 0.0274466060720328 edge_5_2_0_0
   - 0.0702995499436062 edge_5_4_0_0 + 0.0537153638802647 edge_5_6_0_0
   + 0.1189476336982808 edge_6_5_0_0 + 0.1097497412478932 edge_10_1_0_0
   - 0.1391287013673398 edge_10_2_0_0 - 0.1731208710315435 edge_10_3_0_0
   - 0.0960719695112978 edge_10_4_0_0 - 0.1037248215911239 edge_10_6_0_0
   + 3.981312369462407 edge_10_11_0_0 - 0.7091223919450972 edge_10_15_0_0
   + 0.0252994761243104 edge_11_1_0_0 + 1.60966108988466 edge_11_10_0_0
   - 0.1358857584663342 edge_13_1_0_0 - 0.0792883265101941 edge_13_2_0_0
   - 0.00455581529059232 edge_13_3_0_0 - 0.0452160436730608 edge_13_4_0_0
   + 0.1342203782459061 edge_13_5_0_0 - 0.1466554177688303 edge_13_6_0_0
   - 0.4627131951320826 edge_13_10_0_0 - 0.4257392263448618 edge_13_11_0_0
   + 0.0664436694381704 edge_13_14_0_0 + 0.2174125130843916 edge_13_15_0_0
   + 0.3365292375943573 edge_13_17_0_0 + 0.4188412362602539 edge_13_20_0_0
   - 0.1284833133394671 edge_15_2_0_0 + 0.0530862938471901 edge_15_3_0_0
   - 0.1950979860527222 edge_15_4_0_0 + 0.1489780002124538 edge_15_5_0_0
   - 1.175637144111922 edge_15_10_0_0 + 0.2220289107203662 edge_15_13_0_0
   + 0.224531287693623 edge_15_14_0_0 + 0.4552221074934284 edge_15_17_0_0
   + 1.737261395205909 edge_15_20_0_0
Subject To
 one_parent_1: edge_3_1_0_0 + edge_4_1_0_0 + edge_10_1_0_0 + edge_11_1_0_0
   + edge_13_1_0_0 = 1
 consume_one_flow_unit_1: - flow_1_3_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_10_1_0_0 + flow_11_1_0_0 + flow_13_1_0_0 = 1
 one_parent_2: edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0 + edge_10_2_0_0
   + edge_13_2_0_0 + edge_15_2_0_0 = 1
 consume_one_flow_unit_2: - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0
   - flow_2_6_0_0 - flow_2_10_0_0 - flow_2_13_0_0 - flow_2_17_0_0
   - flow_2_20_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_10_2_0_0 + flow_13_2_0_0 + flow_15_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_10_3_0_0
   + edge_13_3_0_0 + edge_15_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_13_0_0
   - flow_3_14_0_0 - flow_3_15_0_0 - flow_3_17_0_0 - flow_3_20_0_0
   + flow_10_3_0_0 + flow_13_3_0_0 + flow_15_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0 + edge_10_4_0_0
   + edge_13_4_0_0 + edge_15_4_0_0 = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 - flow_4_1_0_0
   - flow_4_2_0_0 - flow_4_5_0_0 - flow_4_6_0_0 + flow_5_4_0_0
   + flow_10_4_0_0 + flow_13_4_0_0 + flow_15_4_0_0 = 1
 one_parent_5: edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0 + edge_6_5_0_0
   + edge_13_5_0_0 + edge_15_5_0_0 = 1
 consume_one_flow_unit_5: flow_2_5_0_0 + flow_3_5_0_0 + flow_4_5_0_0
   - flow_5_2_0_0 - flow_5_4_0_0 - flow_5_6_0_0 + flow_6_5_0_0
   + flow_13_5_0_0 + flow_15_5_0_0 = 1
 one_parent_6: edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   + edge_10_6_0_0 + edge_13_6_0_0 = 1
 consume_one_flow_unit_6: flow_2_6_0_0 + flow_3_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_5_0_0 + flow_10_6_0_0 + flow_13_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_11_10_0_0 + edge_13_10_0_0 + edge_15_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_6_0_0 - flow_10_11_0_0 - flow_10_15_0_0 + flow_11_10_0_0
   + flow_13_10_0_0 + flow_15_10_0_0 = 1
 one_parent_11: edge_3_11_0_0 + edge_10_11_0_0 + edge_13_11_0_0 = 1
 consume_one_flow_unit_11: flow_3_11_0_0 + flow_10_11_0_0 - flow_11_1_0_0
   - flow_11_10_0_0 + flow_13_11_0_0 = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: edge_0_13_0_0 + edge_2_13_0_0 + edge_3_13_0_0
   + edge_15_13_0_0 = 1
 consume_one_flow_unit_13: flow_0_13_0_0 + flow_2_13_0_0 + flow_3_13_0_0
   - flow_13_1_0_0 - flow_13_2_0_0 - flow_13_3_0_0 - flow_13_4_0_0
   - flow_13_5_0_0 - flow_13_6_0_0 - flow_13_10_0_0 - flow_13_11_0_0
   - flow_13_14_0_0 - flow_13_15_0_0 - flow_13_17_0_0 - flow_13_20_0_0
   + flow_15_13_0_0 = 1
 one_parent_14: edge_3_14_0_0 + edge_13_14_0_0 + edge_15_14_0_0 = 1
 consume_one_flow_unit_14: flow_3_14_0_0 + flow_13_14_0_0 + flow_15_14_0_0
   = 1
 one_parent_15: edge_0_15_0_0 + edge_3_15_0_0 + edge_10_15_0_0
   + edge_13_15_0_0 = 1
 consume_one_flow_unit_15: flow_0_15_0_0 + flow_3_15_0_0 + flow_10_15_0_0
   + flow_13_15_0_0 - flow_15_2_0_0 - flow_15_3_0_0 - flow_15_4_0_0
   - flow_15_5_0_0 - flow_15_10_0_0 - flow_15_13_0_0 - flow_15_14_0_0
   - flow_15_17_0_0 - flow_15_20_0_0 = 1
 one_parent_16: = 1
 consume_one_flow_unit_16: = 1
 one_parent_17: edge_2_17_0_0 + edge_3_17_0_0 + edge_13_17_0_0
   + edge_15_17_0_0 = 1
 consume_one_flow_unit_17: flow_2_17_0_0 + flow_3_17_0_0 + flow_13_17_0_0
   + flow_15_17_0_0 = 1
 one_parent_18: = 1
 consume_one_flow_unit_18: = 1
 one_parent_19: = 1
 consume_one_flow_unit_19: = 1
 one_parent_20: edge_2_20_0_0 + edge_3_20_0_0 + edge_13_20_0_0
   + edge_15_20_0_0 = 1
 consume_one_flow_unit_20: flow_2_20_0_0 + flow_3_20_0_0 + flow_13_20_0_0
   + flow_15_20_0_0 = 1
 root_flow: flow_0_3_0_0 + flow_0_10_0_0 + flow_0_13_0_0 + flow_0_15_0_0
   = 20
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_13_0_0: flow_0_13_0_0 - 1000 edge_0_13_0_0 <= 0
 inactive_no_flow_0_15_0_0: flow_0_15_0_0 - 1000 edge_0_15_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_2_17_0_0: flow_2_17_0_0 - 1000 edge_2_17_0_0 <= 0
 inactive_no_flow_2_20_0_0: flow_2_20_0_0 - 1000 edge_2_20_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_13_0_0: flow_3_13_0_0 - 1000 edge_3_13_0_0 <= 0
 inactive_no_flow_3_14_0_0: flow_3_14_0_0 - 1000 edge_3_14_0_0 <= 0
 inactive_no_flow_3_15_0_0: flow_3_15_0_0 - 1000 edge_3_15_0_0 <= 0
 inactive_no_flow_3_17_0_0: flow_3_17_0_0 - 1000 edge_3_17_0_0 <= 0
 inactive_no_flow_3_20_0_0: flow_3_20_0_0 - 1000 edge_3_20_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_15_0_0: flow_10_15_0_0 - 1000 edge_10_15_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_13_1_0_0: flow_13_1_0_0 - 1000 edge_13_1_0_0 <= 0
 inactive_no_flow_13_2_0_0: flow_13_2_0_0 - 1000 edge_13_2_0_0 <= 0
 inactive_no_flow_13_3_0_0: flow_13_3_0_0 - 1000 edge_13_3_0_0 <= 0
 inactive_no_flow_13_4_0_0: flow_13_4_0_0 - 1000 edge_13_4_0_0 <= 0
 inactive_no_flow_13_5_0_0: flow_13_5_0_0 - 1000 edge_13_5_0_0 <= 0
 inactive_no_flow_13_6_0_0: flow_13_6_0_0 - 1000 edge_13_6_0_0 <= 0
 inactive_no_flow_13_10_0_0: flow_13_10_0_0 - 1000 edge_13_10_0_0 <= 0
 inactive_no_flow_13_11_0_0: flow_13_11_0_0 - 1000 edge_13_11_0_0 <= 0
 inactive_no_flow_13_14_0_0: flow_13_14_0_0 - 1000 edge_13_14_0_0 <= 0
 inactive_no_flow_13_15_0_0: flow_13_15_0_0 - 1000 edge_13_15_0_0 <= 0
 inactive_no_flow_13_17_0_0: flow_13_17_0_0 - 1000 edge_13_17_0_0 <= 0
 inactive_no_flow_13_20_0_0: flow_13_20_0_0 - 1000 edge_13_20_0_0 <= 0
 inactive_no_flow_15_2_0_0: flow_15_2_0_0 - 1000 edge_15_2_0_0 <= 0
 inactive_no_flow_15_3_0_0: flow_15_3_0_0 - 1000 edge_15_3_0_0 <= 0
 inactive_no_flow_15_4_0_0: flow_15_4_0_0 - 1000 edge_15_4_0_0 <= 0
 inactive_no_flow_15_5_0_0: flow_15_5_0_0 - 1000 edge_15_5_0_0 <= 0
 inactive_no_flow_15_10_0_0: flow_15_10_0_0 - 1000 edge_15_10_0_0 <= 0
 inactive_no_flow_15_13_0_0: flow_15_13_0_0 - 1000 edge_15_13_0_0 <= 0
 inactive_no_flow_15_14_0_0: flow_15_14_0_0 - 1000 edge_15_14_0_0 <= 0
 inactive_no_flow_15_17_0_0: flow_15_17_0_0 - 1000 edge_15_17_0_0 <= 0
 inactive_no_flow_15_20_0_0: flow_15_20_0_0 - 1000 edge_15_20_0_0 <= 0
Bounds
Binaries
 edge_0_3_0_0 edge_0_10_0_0 edge_0_13_0_0 edge_0_15_0_0 edge_1_3_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_10_0_0
 edge_2_13_0_0 edge_2_17_0_0 edge_2_20_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_10_0_0 edge_3_11_0_0
 edge_3_13_0_0 edge_3_14_0_0 edge_3_15_0_0 edge_3_17_0_0 edge_3_20_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_5_2_0_0
 edge_5_4_0_0 edge_5_6_0_0 edge_6_5_0_0 edge_10_1_0_0 edge_10_2_0_0
 edge_10_3_0_0 edge_10_4_0_0 edge_10_6_0_0 edge_10_11_0_0 edge_10_15_0_0
 edge_11_1_0_0 edge_11_10_0_0 edge_13_1_0_0 edge_13_2_0_0 edge_13_3_0_0
 edge_13_4_0_0 edge_13_5_0_0 edge_13_6_0_0 edge_13_10_0_0 edge_13_11_0_0
 edge_13_14_0_0 edge_13_15_0_0 edge_13_17_0_0 edge_13_20_0_0 edge_15_2_0_0
 edge_15_3_0_0 edge_15_4_0_0 edge_15_5_0_0 edge_15_10_0_0 edge_15_13_0_0
 edge_15_14_0_0 edge_15_17_0_0 edge_15_20_0_0
End
