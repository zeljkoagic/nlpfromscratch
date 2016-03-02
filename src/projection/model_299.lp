Maximize
  - 2.7709141649545 edge_0_1_0_10 + 0.101190139652 edge_0_2_0_3
   - 2.3399724635564 edge_0_3_0_9 - 2.934150522525 edge_0_5_0_9
   - 2.487665581017 edge_0_8_0_2 + 0.7330847631100615 edge_1_2_10_3
   + 1.325286543218247 edge_1_3_10_9 + 0.8012306880234521 edge_1_4_10_5
   + 0.1178123258908179 edge_1_5_10_9 - 0.1946382092908937 edge_1_7_10_9
   - 0.6179896574383998 edge_1_8_10_2 - 0.1370786356517535 edge_1_9_10_5
   + 2.525096383543775 edge_2_1_3_10 + 3.357708446266837 edge_2_3_3_9
   + 1.729177379964328 edge_2_4_3_5 + 0.6145595585734976 edge_2_5_3_9
   - 0.0111060865795203 edge_2_6_3_13 + 0.0477220453790472 edge_2_7_3_7
   + 1.059508439429197 edge_2_8_3_2 + 3.640303017745557 edge_2_9_3_5
   + 0.4969455959293269 edge_3_1_9_10 + 0.6160193071091479 edge_3_4_9_5
   + 0.523332286307744 edge_3_5_9_9 - 0.4305786234483817 edge_3_6_9_13
   - 0.8208462658487843 edge_3_7_9_7 - 0.4516109881601951 edge_3_7_9_9
   - 0.5795184913189345 edge_3_8_9_2 + 0.9707736332145205 edge_3_9_9_5
   - 0.5292600399485364 edge_4_1_5_10 - 0.6198622282458066 edge_4_2_5_3
   + 0.2300432583410223 edge_4_3_5_9 - 0.2553165051779279 edge_4_5_5_9
   - 0.6392130688196347 edge_4_8_5_2 + 0.1477813002172226 edge_5_1_9_10
   + 0.1693844712957986 edge_5_2_9_3 + 0.6172125017931035 edge_5_3_9_9
   + 0.6544499707285262 edge_5_4_9_5 + 1.321398117156458 edge_5_6_9_13
   + 0.4804864831888003 edge_5_7_9_7 + 0.2594049095272808 edge_5_7_9_9
   + 1.164954194663874 edge_5_8_9_2 + 1.803280399379709 edge_5_9_9_5
   - 0.9957956150456303 edge_6_4_13_5 - 0.5533143535218028 edge_6_5_13_9
   - 0.3758852554024967 edge_6_7_13_7 - 0.547507094704157 edge_7_1_9_10
   - 0.4358850987886093 edge_7_3_9_9 - 0.0968777763400797 edge_7_4_9_5
   + 0.012582767357845 edge_7_5_7_9 + 0.1836021805808402 edge_7_5_9_9
   + 0.3538993765513225 edge_7_6_7_13 + 0.3946286249640809 edge_7_8_9_2
   + 0.2493210886776897 edge_7_9_9_5 + 0.3120946599817196 edge_8_1_2_10
   + 0.863082728739056 edge_8_2_2_3 + 0.6466875990694571 edge_8_3_2_9
   + 0.4334936179189869 edge_8_4_2_5 + 1.687273280501816 edge_8_5_2_9
   + 2.455690519437028 edge_8_6_2_13 + 3.606947588307194 edge_8_7_2_7
   + 0.8705784535654733 edge_8_7_2_9 + 1.578339477017185 edge_8_9_2_5
Subject To
 one_parent_1: edge_0_1_0_10 + edge_2_1_3_10 + edge_3_1_9_10
   + edge_4_1_5_10 + edge_5_1_9_10 + edge_7_1_9_10 + edge_8_1_2_10 = 1
 one_pos_1: pos_1_10 = 1
 consume_one_flow_unit_1: flow_0_1_0_10 - flow_1_2_10_3 - flow_1_3_10_9
   - flow_1_4_10_5 - flow_1_5_10_9 - flow_1_7_10_9 - flow_1_8_10_2
   - flow_1_9_10_5 + flow_2_1_3_10 + flow_3_1_9_10 + flow_4_1_5_10
   + flow_5_1_9_10 + flow_7_1_9_10 + flow_8_1_2_10 = 1
 one_parent_2: edge_0_2_0_3 + edge_1_2_10_3 + edge_4_2_5_3 + edge_5_2_9_3
   + edge_8_2_2_3 = 1
 one_pos_2: pos_2_3 = 1
 consume_one_flow_unit_2: flow_0_2_0_3 + flow_1_2_10_3 - flow_2_1_3_10
   - flow_2_3_3_9 - flow_2_4_3_5 - flow_2_5_3_9 - flow_2_6_3_13
   - flow_2_7_3_7 - flow_2_8_3_2 - flow_2_9_3_5 + flow_4_2_5_3
   + flow_5_2_9_3 + flow_8_2_2_3 = 1
 one_parent_3: edge_0_3_0_9 + edge_1_3_10_9 + edge_2_3_3_9 + edge_4_3_5_9
   + edge_5_3_9_9 + edge_7_3_9_9 + edge_8_3_2_9 = 1
 one_pos_3: pos_3_9 = 1
 consume_one_flow_unit_3: flow_0_3_0_9 + flow_1_3_10_9 + flow_2_3_3_9
   - flow_3_1_9_10 - flow_3_4_9_5 - flow_3_5_9_9 - flow_3_6_9_13
   - flow_3_7_9_7 - flow_3_7_9_9 - flow_3_8_9_2 - flow_3_9_9_5
   + flow_4_3_5_9 + flow_5_3_9_9 + flow_7_3_9_9 + flow_8_3_2_9 = 1
 one_parent_4: edge_1_4_10_5 + edge_2_4_3_5 + edge_3_4_9_5 + edge_5_4_9_5
   + edge_6_4_13_5 + edge_7_4_9_5 + edge_8_4_2_5 = 1
 one_pos_4: pos_4_5 = 1
 consume_one_flow_unit_4: flow_1_4_10_5 + flow_2_4_3_5 + flow_3_4_9_5
   - flow_4_1_5_10 - flow_4_2_5_3 - flow_4_3_5_9 - flow_4_5_5_9
   - flow_4_8_5_2 + flow_5_4_9_5 + flow_6_4_13_5 + flow_7_4_9_5
   + flow_8_4_2_5 = 1
 one_parent_5: edge_0_5_0_9 + edge_1_5_10_9 + edge_2_5_3_9 + edge_3_5_9_9
   + edge_4_5_5_9 + edge_6_5_13_9 + edge_7_5_7_9 + edge_7_5_9_9
   + edge_8_5_2_9 = 1
 one_pos_5: pos_5_9 = 1
 consume_one_flow_unit_5: flow_0_5_0_9 + flow_1_5_10_9 + flow_2_5_3_9
   + flow_3_5_9_9 + flow_4_5_5_9 - flow_5_1_9_10 - flow_5_2_9_3
   - flow_5_3_9_9 - flow_5_4_9_5 - flow_5_6_9_13 - flow_5_7_9_7
   - flow_5_7_9_9 - flow_5_8_9_2 - flow_5_9_9_5 + flow_6_5_13_9
   + flow_7_5_7_9 + flow_7_5_9_9 + flow_8_5_2_9 = 1
 one_parent_6: edge_2_6_3_13 + edge_3_6_9_13 + edge_5_6_9_13
   + edge_7_6_7_13 + edge_8_6_2_13 = 1
 one_pos_6: pos_6_13 = 1
 consume_one_flow_unit_6: flow_2_6_3_13 + flow_3_6_9_13 + flow_5_6_9_13
   - flow_6_4_13_5 - flow_6_5_13_9 - flow_6_7_13_7 + flow_7_6_7_13
   + flow_8_6_2_13 = 1
 one_parent_7: edge_1_7_10_9 + edge_2_7_3_7 + edge_3_7_9_7 + edge_3_7_9_9
   + edge_5_7_9_7 + edge_5_7_9_9 + edge_6_7_13_7 + edge_8_7_2_7
   + edge_8_7_2_9 = 1
 one_pos_7: pos_7_9 + pos_7_7 = 1
 consume_one_flow_unit_7: flow_1_7_10_9 + flow_2_7_3_7 + flow_3_7_9_7
   + flow_3_7_9_9 + flow_5_7_9_7 + flow_5_7_9_9 + flow_6_7_13_7
   - flow_7_1_9_10 - flow_7_3_9_9 - flow_7_4_9_5 - flow_7_5_7_9
   - flow_7_5_9_9 - flow_7_6_7_13 - flow_7_8_9_2 - flow_7_9_9_5
   + flow_8_7_2_7 + flow_8_7_2_9 = 1
 one_parent_8: edge_0_8_0_2 + edge_1_8_10_2 + edge_2_8_3_2 + edge_3_8_9_2
   + edge_4_8_5_2 + edge_5_8_9_2 + edge_7_8_9_2 = 1
 one_pos_8: pos_8_2 = 1
 consume_one_flow_unit_8: flow_0_8_0_2 + flow_1_8_10_2 + flow_2_8_3_2
   + flow_3_8_9_2 + flow_4_8_5_2 + flow_5_8_9_2 + flow_7_8_9_2
   - flow_8_1_2_10 - flow_8_2_2_3 - flow_8_3_2_9 - flow_8_4_2_5
   - flow_8_5_2_9 - flow_8_6_2_13 - flow_8_7_2_7 - flow_8_7_2_9
   - flow_8_9_2_5 = 1
 one_parent_9: edge_1_9_10_5 + edge_2_9_3_5 + edge_3_9_9_5 + edge_5_9_9_5
   + edge_7_9_9_5 + edge_8_9_2_5 = 1
 one_pos_9: pos_9_5 = 1
 consume_one_flow_unit_9: flow_1_9_10_5 + flow_2_9_3_5 + flow_3_9_9_5
   + flow_5_9_9_5 + flow_7_9_9_5 + flow_8_9_2_5 = 1
 root_flow: flow_0_1_0_10 + flow_0_2_0_3 + flow_0_3_0_9 + flow_0_5_0_9
   + flow_0_8_0_2 = 9
 inactive_no_flow_0_1_0_10: flow_0_1_0_10 - 1000 edge_0_1_0_10 <= 0
 inactive_no_flow_0_2_0_3: flow_0_2_0_3 - 1000 edge_0_2_0_3 <= 0
 inactive_no_flow_0_3_0_9: flow_0_3_0_9 - 1000 edge_0_3_0_9 <= 0
 inactive_no_flow_0_5_0_9: flow_0_5_0_9 - 1000 edge_0_5_0_9 <= 0
 inactive_no_flow_0_8_0_2: flow_0_8_0_2 - 1000 edge_0_8_0_2 <= 0
 inactive_no_flow_1_2_10_3: flow_1_2_10_3 - 1000 edge_1_2_10_3 <= 0
 inactive_no_flow_1_3_10_9: flow_1_3_10_9 - 1000 edge_1_3_10_9 <= 0
 inactive_no_flow_1_4_10_5: flow_1_4_10_5 - 1000 edge_1_4_10_5 <= 0
 inactive_no_flow_1_5_10_9: flow_1_5_10_9 - 1000 edge_1_5_10_9 <= 0
 inactive_no_flow_1_7_10_9: flow_1_7_10_9 - 1000 edge_1_7_10_9 <= 0
 inactive_no_flow_1_8_10_2: flow_1_8_10_2 - 1000 edge_1_8_10_2 <= 0
 inactive_no_flow_1_9_10_5: flow_1_9_10_5 - 1000 edge_1_9_10_5 <= 0
 inactive_no_flow_2_1_3_10: flow_2_1_3_10 - 1000 edge_2_1_3_10 <= 0
 inactive_no_flow_2_3_3_9: flow_2_3_3_9 - 1000 edge_2_3_3_9 <= 0
 inactive_no_flow_2_4_3_5: flow_2_4_3_5 - 1000 edge_2_4_3_5 <= 0
 inactive_no_flow_2_5_3_9: flow_2_5_3_9 - 1000 edge_2_5_3_9 <= 0
 inactive_no_flow_2_6_3_13: flow_2_6_3_13 - 1000 edge_2_6_3_13 <= 0
 inactive_no_flow_2_7_3_7: flow_2_7_3_7 - 1000 edge_2_7_3_7 <= 0
 inactive_no_flow_2_8_3_2: flow_2_8_3_2 - 1000 edge_2_8_3_2 <= 0
 inactive_no_flow_2_9_3_5: flow_2_9_3_5 - 1000 edge_2_9_3_5 <= 0
 inactive_no_flow_3_1_9_10: flow_3_1_9_10 - 1000 edge_3_1_9_10 <= 0
 inactive_no_flow_3_4_9_5: flow_3_4_9_5 - 1000 edge_3_4_9_5 <= 0
 inactive_no_flow_3_5_9_9: flow_3_5_9_9 - 1000 edge_3_5_9_9 <= 0
 inactive_no_flow_3_6_9_13: flow_3_6_9_13 - 1000 edge_3_6_9_13 <= 0
 inactive_no_flow_3_7_9_7: flow_3_7_9_7 - 1000 edge_3_7_9_7 <= 0
 inactive_no_flow_3_7_9_9: flow_3_7_9_9 - 1000 edge_3_7_9_9 <= 0
 inactive_no_flow_3_8_9_2: flow_3_8_9_2 - 1000 edge_3_8_9_2 <= 0
 inactive_no_flow_3_9_9_5: flow_3_9_9_5 - 1000 edge_3_9_9_5 <= 0
 inactive_no_flow_4_1_5_10: flow_4_1_5_10 - 1000 edge_4_1_5_10 <= 0
 inactive_no_flow_4_2_5_3: flow_4_2_5_3 - 1000 edge_4_2_5_3 <= 0
 inactive_no_flow_4_3_5_9: flow_4_3_5_9 - 1000 edge_4_3_5_9 <= 0
 inactive_no_flow_4_5_5_9: flow_4_5_5_9 - 1000 edge_4_5_5_9 <= 0
 inactive_no_flow_4_8_5_2: flow_4_8_5_2 - 1000 edge_4_8_5_2 <= 0
 inactive_no_flow_5_1_9_10: flow_5_1_9_10 - 1000 edge_5_1_9_10 <= 0
 inactive_no_flow_5_2_9_3: flow_5_2_9_3 - 1000 edge_5_2_9_3 <= 0
 inactive_no_flow_5_3_9_9: flow_5_3_9_9 - 1000 edge_5_3_9_9 <= 0
 inactive_no_flow_5_4_9_5: flow_5_4_9_5 - 1000 edge_5_4_9_5 <= 0
 inactive_no_flow_5_6_9_13: flow_5_6_9_13 - 1000 edge_5_6_9_13 <= 0
 inactive_no_flow_5_7_9_7: flow_5_7_9_7 - 1000 edge_5_7_9_7 <= 0
 inactive_no_flow_5_7_9_9: flow_5_7_9_9 - 1000 edge_5_7_9_9 <= 0
 inactive_no_flow_5_8_9_2: flow_5_8_9_2 - 1000 edge_5_8_9_2 <= 0
 inactive_no_flow_5_9_9_5: flow_5_9_9_5 - 1000 edge_5_9_9_5 <= 0
 inactive_no_flow_6_4_13_5: flow_6_4_13_5 - 1000 edge_6_4_13_5 <= 0
 inactive_no_flow_6_5_13_9: flow_6_5_13_9 - 1000 edge_6_5_13_9 <= 0
 inactive_no_flow_6_7_13_7: flow_6_7_13_7 - 1000 edge_6_7_13_7 <= 0
 inactive_no_flow_7_1_9_10: flow_7_1_9_10 - 1000 edge_7_1_9_10 <= 0
 inactive_no_flow_7_3_9_9: flow_7_3_9_9 - 1000 edge_7_3_9_9 <= 0
 inactive_no_flow_7_4_9_5: flow_7_4_9_5 - 1000 edge_7_4_9_5 <= 0
 inactive_no_flow_7_5_7_9: flow_7_5_7_9 - 1000 edge_7_5_7_9 <= 0
 inactive_no_flow_7_5_9_9: flow_7_5_9_9 - 1000 edge_7_5_9_9 <= 0
 inactive_no_flow_7_6_7_13: flow_7_6_7_13 - 1000 edge_7_6_7_13 <= 0
 inactive_no_flow_7_8_9_2: flow_7_8_9_2 - 1000 edge_7_8_9_2 <= 0
 inactive_no_flow_7_9_9_5: flow_7_9_9_5 - 1000 edge_7_9_9_5 <= 0
 inactive_no_flow_8_1_2_10: flow_8_1_2_10 - 1000 edge_8_1_2_10 <= 0
 inactive_no_flow_8_2_2_3: flow_8_2_2_3 - 1000 edge_8_2_2_3 <= 0
 inactive_no_flow_8_3_2_9: flow_8_3_2_9 - 1000 edge_8_3_2_9 <= 0
 inactive_no_flow_8_4_2_5: flow_8_4_2_5 - 1000 edge_8_4_2_5 <= 0
 inactive_no_flow_8_5_2_9: flow_8_5_2_9 - 1000 edge_8_5_2_9 <= 0
 inactive_no_flow_8_6_2_13: flow_8_6_2_13 - 1000 edge_8_6_2_13 <= 0
 inactive_no_flow_8_7_2_7: flow_8_7_2_7 - 1000 edge_8_7_2_7 <= 0
 inactive_no_flow_8_7_2_9: flow_8_7_2_9 - 1000 edge_8_7_2_9 <= 0
 inactive_no_flow_8_9_2_5: flow_8_9_2_5 - 1000 edge_8_9_2_5 <= 0
 arc_0_1_0_10_pos_must_match_v: - edge_0_1_0_10 + pos_1_10 >= 0
 arc_0_2_0_3_pos_must_match_v: - edge_0_2_0_3 + pos_2_3 >= 0
 arc_0_3_0_9_pos_must_match_v: - edge_0_3_0_9 + pos_3_9 >= 0
 arc_0_5_0_9_pos_must_match_v: - edge_0_5_0_9 + pos_5_9 >= 0
 arc_0_8_0_2_pos_must_match_v: - edge_0_8_0_2 + pos_8_2 >= 0
 arc_1_2_10_3_pos_must_match_u: - edge_1_2_10_3 + pos_1_10 >= 0
 arc_1_2_10_3_pos_must_match_v: - edge_1_2_10_3 + pos_2_3 >= 0
 arc_1_3_10_9_pos_must_match_u: - edge_1_3_10_9 + pos_1_10 >= 0
 arc_1_3_10_9_pos_must_match_v: - edge_1_3_10_9 + pos_3_9 >= 0
 arc_1_4_10_5_pos_must_match_u: - edge_1_4_10_5 + pos_1_10 >= 0
 arc_1_4_10_5_pos_must_match_v: - edge_1_4_10_5 + pos_4_5 >= 0
 arc_1_5_10_9_pos_must_match_u: - edge_1_5_10_9 + pos_1_10 >= 0
 arc_1_5_10_9_pos_must_match_v: - edge_1_5_10_9 + pos_5_9 >= 0
 arc_1_7_10_9_pos_must_match_u: - edge_1_7_10_9 + pos_1_10 >= 0
 arc_1_7_10_9_pos_must_match_v: - edge_1_7_10_9 + pos_7_9 >= 0
 arc_1_8_10_2_pos_must_match_u: - edge_1_8_10_2 + pos_1_10 >= 0
 arc_1_8_10_2_pos_must_match_v: - edge_1_8_10_2 + pos_8_2 >= 0
 arc_1_9_10_5_pos_must_match_u: - edge_1_9_10_5 + pos_1_10 >= 0
 arc_1_9_10_5_pos_must_match_v: - edge_1_9_10_5 + pos_9_5 >= 0
 arc_2_1_3_10_pos_must_match_u: - edge_2_1_3_10 + pos_2_3 >= 0
 arc_2_1_3_10_pos_must_match_v: - edge_2_1_3_10 + pos_1_10 >= 0
 arc_2_3_3_9_pos_must_match_u: - edge_2_3_3_9 + pos_2_3 >= 0
 arc_2_3_3_9_pos_must_match_v: - edge_2_3_3_9 + pos_3_9 >= 0
 arc_2_4_3_5_pos_must_match_u: - edge_2_4_3_5 + pos_2_3 >= 0
 arc_2_4_3_5_pos_must_match_v: - edge_2_4_3_5 + pos_4_5 >= 0
 arc_2_5_3_9_pos_must_match_u: - edge_2_5_3_9 + pos_2_3 >= 0
 arc_2_5_3_9_pos_must_match_v: - edge_2_5_3_9 + pos_5_9 >= 0
 arc_2_6_3_13_pos_must_match_u: - edge_2_6_3_13 + pos_2_3 >= 0
 arc_2_6_3_13_pos_must_match_v: - edge_2_6_3_13 + pos_6_13 >= 0
 arc_2_7_3_7_pos_must_match_u: - edge_2_7_3_7 + pos_2_3 >= 0
 arc_2_7_3_7_pos_must_match_v: - edge_2_7_3_7 + pos_7_7 >= 0
 arc_2_8_3_2_pos_must_match_u: - edge_2_8_3_2 + pos_2_3 >= 0
 arc_2_8_3_2_pos_must_match_v: - edge_2_8_3_2 + pos_8_2 >= 0
 arc_2_9_3_5_pos_must_match_u: - edge_2_9_3_5 + pos_2_3 >= 0
 arc_2_9_3_5_pos_must_match_v: - edge_2_9_3_5 + pos_9_5 >= 0
 arc_3_1_9_10_pos_must_match_u: - edge_3_1_9_10 + pos_3_9 >= 0
 arc_3_1_9_10_pos_must_match_v: - edge_3_1_9_10 + pos_1_10 >= 0
 arc_3_4_9_5_pos_must_match_u: - edge_3_4_9_5 + pos_3_9 >= 0
 arc_3_4_9_5_pos_must_match_v: - edge_3_4_9_5 + pos_4_5 >= 0
 arc_3_5_9_9_pos_must_match_u: - edge_3_5_9_9 + pos_3_9 >= 0
 arc_3_5_9_9_pos_must_match_v: - edge_3_5_9_9 + pos_5_9 >= 0
 arc_3_6_9_13_pos_must_match_u: - edge_3_6_9_13 + pos_3_9 >= 0
 arc_3_6_9_13_pos_must_match_v: - edge_3_6_9_13 + pos_6_13 >= 0
 arc_3_7_9_7_pos_must_match_u: - edge_3_7_9_7 + pos_3_9 >= 0
 arc_3_7_9_7_pos_must_match_v: - edge_3_7_9_7 + pos_7_7 >= 0
 arc_3_7_9_9_pos_must_match_u: - edge_3_7_9_9 + pos_3_9 >= 0
 arc_3_7_9_9_pos_must_match_v: - edge_3_7_9_9 + pos_7_9 >= 0
 arc_3_8_9_2_pos_must_match_u: - edge_3_8_9_2 + pos_3_9 >= 0
 arc_3_8_9_2_pos_must_match_v: - edge_3_8_9_2 + pos_8_2 >= 0
 arc_3_9_9_5_pos_must_match_u: - edge_3_9_9_5 + pos_3_9 >= 0
 arc_3_9_9_5_pos_must_match_v: - edge_3_9_9_5 + pos_9_5 >= 0
 arc_4_1_5_10_pos_must_match_u: - edge_4_1_5_10 + pos_4_5 >= 0
 arc_4_1_5_10_pos_must_match_v: - edge_4_1_5_10 + pos_1_10 >= 0
 arc_4_2_5_3_pos_must_match_u: - edge_4_2_5_3 + pos_4_5 >= 0
 arc_4_2_5_3_pos_must_match_v: - edge_4_2_5_3 + pos_2_3 >= 0
 arc_4_3_5_9_pos_must_match_u: - edge_4_3_5_9 + pos_4_5 >= 0
 arc_4_3_5_9_pos_must_match_v: - edge_4_3_5_9 + pos_3_9 >= 0
 arc_4_5_5_9_pos_must_match_u: - edge_4_5_5_9 + pos_4_5 >= 0
 arc_4_5_5_9_pos_must_match_v: - edge_4_5_5_9 + pos_5_9 >= 0
 arc_4_8_5_2_pos_must_match_u: - edge_4_8_5_2 + pos_4_5 >= 0
 arc_4_8_5_2_pos_must_match_v: - edge_4_8_5_2 + pos_8_2 >= 0
 arc_5_1_9_10_pos_must_match_u: - edge_5_1_9_10 + pos_5_9 >= 0
 arc_5_1_9_10_pos_must_match_v: - edge_5_1_9_10 + pos_1_10 >= 0
 arc_5_2_9_3_pos_must_match_u: - edge_5_2_9_3 + pos_5_9 >= 0
 arc_5_2_9_3_pos_must_match_v: - edge_5_2_9_3 + pos_2_3 >= 0
 arc_5_3_9_9_pos_must_match_u: - edge_5_3_9_9 + pos_5_9 >= 0
 arc_5_3_9_9_pos_must_match_v: - edge_5_3_9_9 + pos_3_9 >= 0
 arc_5_4_9_5_pos_must_match_u: - edge_5_4_9_5 + pos_5_9 >= 0
 arc_5_4_9_5_pos_must_match_v: - edge_5_4_9_5 + pos_4_5 >= 0
 arc_5_6_9_13_pos_must_match_u: - edge_5_6_9_13 + pos_5_9 >= 0
 arc_5_6_9_13_pos_must_match_v: - edge_5_6_9_13 + pos_6_13 >= 0
 arc_5_7_9_7_pos_must_match_u: - edge_5_7_9_7 + pos_5_9 >= 0
 arc_5_7_9_7_pos_must_match_v: - edge_5_7_9_7 + pos_7_7 >= 0
 arc_5_7_9_9_pos_must_match_u: - edge_5_7_9_9 + pos_5_9 >= 0
 arc_5_7_9_9_pos_must_match_v: - edge_5_7_9_9 + pos_7_9 >= 0
 arc_5_8_9_2_pos_must_match_u: - edge_5_8_9_2 + pos_5_9 >= 0
 arc_5_8_9_2_pos_must_match_v: - edge_5_8_9_2 + pos_8_2 >= 0
 arc_5_9_9_5_pos_must_match_u: - edge_5_9_9_5 + pos_5_9 >= 0
 arc_5_9_9_5_pos_must_match_v: - edge_5_9_9_5 + pos_9_5 >= 0
 arc_6_4_13_5_pos_must_match_u: - edge_6_4_13_5 + pos_6_13 >= 0
 arc_6_4_13_5_pos_must_match_v: - edge_6_4_13_5 + pos_4_5 >= 0
 arc_6_5_13_9_pos_must_match_u: - edge_6_5_13_9 + pos_6_13 >= 0
 arc_6_5_13_9_pos_must_match_v: - edge_6_5_13_9 + pos_5_9 >= 0
 arc_6_7_13_7_pos_must_match_u: - edge_6_7_13_7 + pos_6_13 >= 0
 arc_6_7_13_7_pos_must_match_v: - edge_6_7_13_7 + pos_7_7 >= 0
 arc_7_1_9_10_pos_must_match_u: - edge_7_1_9_10 + pos_7_9 >= 0
 arc_7_1_9_10_pos_must_match_v: - edge_7_1_9_10 + pos_1_10 >= 0
 arc_7_3_9_9_pos_must_match_u: - edge_7_3_9_9 + pos_7_9 >= 0
 arc_7_3_9_9_pos_must_match_v: - edge_7_3_9_9 + pos_3_9 >= 0
 arc_7_4_9_5_pos_must_match_u: - edge_7_4_9_5 + pos_7_9 >= 0
 arc_7_4_9_5_pos_must_match_v: - edge_7_4_9_5 + pos_4_5 >= 0
 arc_7_5_7_9_pos_must_match_u: - edge_7_5_7_9 + pos_7_7 >= 0
 arc_7_5_7_9_pos_must_match_v: - edge_7_5_7_9 + pos_5_9 >= 0
 arc_7_5_9_9_pos_must_match_u: - edge_7_5_9_9 + pos_7_9 >= 0
 arc_7_5_9_9_pos_must_match_v: - edge_7_5_9_9 + pos_5_9 >= 0
 arc_7_6_7_13_pos_must_match_u: - edge_7_6_7_13 + pos_7_7 >= 0
 arc_7_6_7_13_pos_must_match_v: - edge_7_6_7_13 + pos_6_13 >= 0
 arc_7_8_9_2_pos_must_match_u: - edge_7_8_9_2 + pos_7_9 >= 0
 arc_7_8_9_2_pos_must_match_v: - edge_7_8_9_2 + pos_8_2 >= 0
 arc_7_9_9_5_pos_must_match_u: - edge_7_9_9_5 + pos_7_9 >= 0
 arc_7_9_9_5_pos_must_match_v: - edge_7_9_9_5 + pos_9_5 >= 0
 arc_8_1_2_10_pos_must_match_u: - edge_8_1_2_10 + pos_8_2 >= 0
 arc_8_1_2_10_pos_must_match_v: - edge_8_1_2_10 + pos_1_10 >= 0
 arc_8_2_2_3_pos_must_match_u: - edge_8_2_2_3 + pos_8_2 >= 0
 arc_8_2_2_3_pos_must_match_v: - edge_8_2_2_3 + pos_2_3 >= 0
 arc_8_3_2_9_pos_must_match_u: - edge_8_3_2_9 + pos_8_2 >= 0
 arc_8_3_2_9_pos_must_match_v: - edge_8_3_2_9 + pos_3_9 >= 0
 arc_8_4_2_5_pos_must_match_u: - edge_8_4_2_5 + pos_8_2 >= 0
 arc_8_4_2_5_pos_must_match_v: - edge_8_4_2_5 + pos_4_5 >= 0
 arc_8_5_2_9_pos_must_match_u: - edge_8_5_2_9 + pos_8_2 >= 0
 arc_8_5_2_9_pos_must_match_v: - edge_8_5_2_9 + pos_5_9 >= 0
 arc_8_6_2_13_pos_must_match_u: - edge_8_6_2_13 + pos_8_2 >= 0
 arc_8_6_2_13_pos_must_match_v: - edge_8_6_2_13 + pos_6_13 >= 0
 arc_8_7_2_7_pos_must_match_u: - edge_8_7_2_7 + pos_8_2 >= 0
 arc_8_7_2_7_pos_must_match_v: - edge_8_7_2_7 + pos_7_7 >= 0
 arc_8_7_2_9_pos_must_match_u: - edge_8_7_2_9 + pos_8_2 >= 0
 arc_8_7_2_9_pos_must_match_v: - edge_8_7_2_9 + pos_7_9 >= 0
 arc_8_9_2_5_pos_must_match_u: - edge_8_9_2_5 + pos_8_2 >= 0
 arc_8_9_2_5_pos_must_match_v: - edge_8_9_2_5 + pos_9_5 >= 0
Bounds
Binaries
 edge_0_1_0_10 edge_0_2_0_3 edge_0_3_0_9 edge_0_5_0_9 edge_0_8_0_2
 edge_1_2_10_3 edge_1_3_10_9 edge_1_4_10_5 edge_1_5_10_9 edge_1_7_10_9
 edge_1_8_10_2 edge_1_9_10_5 edge_2_1_3_10 edge_2_3_3_9 edge_2_4_3_5
 edge_2_5_3_9 edge_2_6_3_13 edge_2_7_3_7 edge_2_8_3_2 edge_2_9_3_5
 edge_3_1_9_10 edge_3_4_9_5 edge_3_5_9_9 edge_3_6_9_13 edge_3_7_9_7
 edge_3_7_9_9 edge_3_8_9_2 edge_3_9_9_5 edge_4_1_5_10 edge_4_2_5_3
 edge_4_3_5_9 edge_4_5_5_9 edge_4_8_5_2 edge_5_1_9_10 edge_5_2_9_3
 edge_5_3_9_9 edge_5_4_9_5 edge_5_6_9_13 edge_5_7_9_7 edge_5_7_9_9
 edge_5_8_9_2 edge_5_9_9_5 edge_6_4_13_5 edge_6_5_13_9 edge_6_7_13_7
 edge_7_1_9_10 edge_7_3_9_9 edge_7_4_9_5 edge_7_5_7_9 edge_7_5_9_9
 edge_7_6_7_13 edge_7_8_9_2 edge_7_9_9_5 edge_8_1_2_10 edge_8_2_2_3
 edge_8_3_2_9 edge_8_4_2_5 edge_8_5_2_9 edge_8_6_2_13 edge_8_7_2_7
 edge_8_7_2_9 edge_8_9_2_5 pos_1_10 pos_2_3 pos_3_9 pos_4_5 pos_5_9
 pos_6_13 pos_7_9 pos_7_7 pos_8_2 pos_9_5
End
