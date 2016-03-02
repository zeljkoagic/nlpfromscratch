Maximize
  - 3.6566297967528 edge_0_1_0_4 - 2.7659527492948 edge_0_3_0_2
   - 0.2073944752218 edge_0_4_0_3 - 2.569573634571 edge_0_5_0_10
   - 3.2745724320046 edge_0_8_0_2 - 0.8386525586423016 edge_1_3_4_2
   - 0.2906362173373659 edge_1_4_4_3 - 0.6374005250553192 edge_1_5_4_10
   - 0.8780917585711656 edge_1_6_4_1 - 1.331440936084046 edge_1_8_4_2
   - 0.3122229429604197 edge_1_9_4_5 - 0.1819847450968532 edge_2_4_7_3
   - 1.233948899903183 edge_2_5_7_10 - 1.450027882546474 edge_2_8_7_2
   - 0.8302424502554869 edge_3_1_2_4 + 3.545016331524 edge_3_2_2_7
   + 0.9736720656966401 edge_3_4_2_3 + 0.5814687410921189 edge_3_5_2_10
   + 0.2863415153246818 edge_3_6_2_1 + 0.9986347805383182 edge_3_8_2_2
   + 1.182177027109261 edge_3_9_2_5 + 0.7121862608375854 edge_4_1_3_4
   + 1.338851651523426 edge_4_2_3_7 + 1.6876526900936 edge_4_3_3_2
   + 2.597458658890711 edge_4_5_3_10 + 0.4355771856746158 edge_4_6_3_1
   + 0.1419359733414445 edge_4_7_3_7 + 1.8039432042942 edge_4_8_3_2
   + 2.260837814040521 edge_4_9_3_5 - 0.2879327828945916 edge_5_1_10_4
   + 0.5202307615713493 edge_5_3_10_2 + 0.1958871913287241 edge_5_4_10_3
   + 0.4707174169490927 edge_5_6_10_1 + 0.0187424883706623 edge_5_7_10_7
   + 1.078017121539829 edge_5_8_10_2 + 0.4555999215460095 edge_5_9_10_5
   - 0.796250297634243 edge_6_1_1_4 - 0.4965599659330585 edge_6_3_1_2
   + 0.2810439570968133 edge_6_5_1_10 + 0.478976311566184 edge_6_7_1_7
   + 1.061453771427444 edge_6_8_1_2 - 0.0282743832472088 edge_6_9_1_5
   - 1.056903487876667 edge_7_5_7_10 - 0.571601092302267 edge_7_6_7_1
   - 1.258784540799931 edge_8_1_2_4 - 0.083247328022268 edge_8_3_2_2
   - 0.0569812183664437 edge_8_4_2_3 + 1.359568224871008 edge_8_5_2_10
   + 2.909377589569514 edge_8_6_2_1 + 3.631642740315168 edge_8_7_2_7
   + 0.2043152977416596 edge_8_9_2_5 - 2.34021542157539 edge_9_1_5_4
   - 1.140931329412749 edge_9_5_5_10 - 0.9562676879333697 edge_9_6_5_1
   - 1.666689799049422 edge_9_8_5_2
Subject To
 one_parent_1: edge_0_1_0_4 + edge_3_1_2_4 + edge_4_1_3_4 + edge_5_1_10_4
   + edge_6_1_1_4 + edge_8_1_2_4 + edge_9_1_5_4 = 1
 one_pos_1: pos_1_4 = 1
 consume_one_flow_unit_1: flow_0_1_0_4 - flow_1_3_4_2 - flow_1_4_4_3
   - flow_1_5_4_10 - flow_1_6_4_1 - flow_1_8_4_2 - flow_1_9_4_5
   + flow_3_1_2_4 + flow_4_1_3_4 + flow_5_1_10_4 + flow_6_1_1_4
   + flow_8_1_2_4 + flow_9_1_5_4 = 1
 one_parent_2: edge_3_2_2_7 + edge_4_2_3_7 = 1
 one_pos_2: pos_2_7 = 1
 consume_one_flow_unit_2: - flow_2_4_7_3 - flow_2_5_7_10 - flow_2_8_7_2
   + flow_3_2_2_7 + flow_4_2_3_7 = 1
 one_parent_3: edge_0_3_0_2 + edge_1_3_4_2 + edge_4_3_3_2 + edge_5_3_10_2
   + edge_6_3_1_2 + edge_8_3_2_2 = 1
 one_pos_3: pos_3_2 = 1
 consume_one_flow_unit_3: flow_0_3_0_2 + flow_1_3_4_2 - flow_3_1_2_4
   - flow_3_2_2_7 - flow_3_4_2_3 - flow_3_5_2_10 - flow_3_6_2_1
   - flow_3_8_2_2 - flow_3_9_2_5 + flow_4_3_3_2 + flow_5_3_10_2
   + flow_6_3_1_2 + flow_8_3_2_2 = 1
 one_parent_4: edge_0_4_0_3 + edge_1_4_4_3 + edge_2_4_7_3 + edge_3_4_2_3
   + edge_5_4_10_3 + edge_8_4_2_3 = 1
 one_pos_4: pos_4_3 = 1
 consume_one_flow_unit_4: flow_0_4_0_3 + flow_1_4_4_3 + flow_2_4_7_3
   + flow_3_4_2_3 - flow_4_1_3_4 - flow_4_2_3_7 - flow_4_3_3_2
   - flow_4_5_3_10 - flow_4_6_3_1 - flow_4_7_3_7 - flow_4_8_3_2
   - flow_4_9_3_5 + flow_5_4_10_3 + flow_8_4_2_3 = 1
 one_parent_5: edge_0_5_0_10 + edge_1_5_4_10 + edge_2_5_7_10
   + edge_3_5_2_10 + edge_4_5_3_10 + edge_6_5_1_10 + edge_7_5_7_10
   + edge_8_5_2_10 + edge_9_5_5_10 = 1
 one_pos_5: pos_5_10 = 1
 consume_one_flow_unit_5: flow_0_5_0_10 + flow_1_5_4_10 + flow_2_5_7_10
   + flow_3_5_2_10 + flow_4_5_3_10 - flow_5_1_10_4 - flow_5_3_10_2
   - flow_5_4_10_3 - flow_5_6_10_1 - flow_5_7_10_7 - flow_5_8_10_2
   - flow_5_9_10_5 + flow_6_5_1_10 + flow_7_5_7_10 + flow_8_5_2_10
   + flow_9_5_5_10 = 1
 one_parent_6: edge_1_6_4_1 + edge_3_6_2_1 + edge_4_6_3_1 + edge_5_6_10_1
   + edge_7_6_7_1 + edge_8_6_2_1 + edge_9_6_5_1 = 1
 one_pos_6: pos_6_1 = 1
 consume_one_flow_unit_6: flow_1_6_4_1 + flow_3_6_2_1 + flow_4_6_3_1
   + flow_5_6_10_1 - flow_6_1_1_4 - flow_6_3_1_2 - flow_6_5_1_10
   - flow_6_7_1_7 - flow_6_8_1_2 - flow_6_9_1_5 + flow_7_6_7_1
   + flow_8_6_2_1 + flow_9_6_5_1 = 1
 one_parent_7: edge_4_7_3_7 + edge_5_7_10_7 + edge_6_7_1_7 + edge_8_7_2_7
   = 1
 one_pos_7: pos_7_7 = 1
 consume_one_flow_unit_7: flow_4_7_3_7 + flow_5_7_10_7 + flow_6_7_1_7
   - flow_7_5_7_10 - flow_7_6_7_1 + flow_8_7_2_7 = 1
 one_parent_8: edge_0_8_0_2 + edge_1_8_4_2 + edge_2_8_7_2 + edge_3_8_2_2
   + edge_4_8_3_2 + edge_5_8_10_2 + edge_6_8_1_2 + edge_9_8_5_2 = 1
 one_pos_8: pos_8_2 = 1
 consume_one_flow_unit_8: flow_0_8_0_2 + flow_1_8_4_2 + flow_2_8_7_2
   + flow_3_8_2_2 + flow_4_8_3_2 + flow_5_8_10_2 + flow_6_8_1_2
   - flow_8_1_2_4 - flow_8_3_2_2 - flow_8_4_2_3 - flow_8_5_2_10
   - flow_8_6_2_1 - flow_8_7_2_7 - flow_8_9_2_5 + flow_9_8_5_2 = 1
 one_parent_9: edge_1_9_4_5 + edge_3_9_2_5 + edge_4_9_3_5 + edge_5_9_10_5
   + edge_6_9_1_5 + edge_8_9_2_5 = 1
 one_pos_9: pos_9_5 = 1
 consume_one_flow_unit_9: flow_1_9_4_5 + flow_3_9_2_5 + flow_4_9_3_5
   + flow_5_9_10_5 + flow_6_9_1_5 + flow_8_9_2_5 - flow_9_1_5_4
   - flow_9_5_5_10 - flow_9_6_5_1 - flow_9_8_5_2 = 1
 root_flow: flow_0_1_0_4 + flow_0_3_0_2 + flow_0_4_0_3 + flow_0_5_0_10
   + flow_0_8_0_2 = 9
 inactive_no_flow_0_1_0_4: flow_0_1_0_4 - 1000 edge_0_1_0_4 <= 0
 inactive_no_flow_0_3_0_2: flow_0_3_0_2 - 1000 edge_0_3_0_2 <= 0
 inactive_no_flow_0_4_0_3: flow_0_4_0_3 - 1000 edge_0_4_0_3 <= 0
 inactive_no_flow_0_5_0_10: flow_0_5_0_10 - 1000 edge_0_5_0_10 <= 0
 inactive_no_flow_0_8_0_2: flow_0_8_0_2 - 1000 edge_0_8_0_2 <= 0
 inactive_no_flow_1_3_4_2: flow_1_3_4_2 - 1000 edge_1_3_4_2 <= 0
 inactive_no_flow_1_4_4_3: flow_1_4_4_3 - 1000 edge_1_4_4_3 <= 0
 inactive_no_flow_1_5_4_10: flow_1_5_4_10 - 1000 edge_1_5_4_10 <= 0
 inactive_no_flow_1_6_4_1: flow_1_6_4_1 - 1000 edge_1_6_4_1 <= 0
 inactive_no_flow_1_8_4_2: flow_1_8_4_2 - 1000 edge_1_8_4_2 <= 0
 inactive_no_flow_1_9_4_5: flow_1_9_4_5 - 1000 edge_1_9_4_5 <= 0
 inactive_no_flow_2_4_7_3: flow_2_4_7_3 - 1000 edge_2_4_7_3 <= 0
 inactive_no_flow_2_5_7_10: flow_2_5_7_10 - 1000 edge_2_5_7_10 <= 0
 inactive_no_flow_2_8_7_2: flow_2_8_7_2 - 1000 edge_2_8_7_2 <= 0
 inactive_no_flow_3_1_2_4: flow_3_1_2_4 - 1000 edge_3_1_2_4 <= 0
 inactive_no_flow_3_2_2_7: flow_3_2_2_7 - 1000 edge_3_2_2_7 <= 0
 inactive_no_flow_3_4_2_3: flow_3_4_2_3 - 1000 edge_3_4_2_3 <= 0
 inactive_no_flow_3_5_2_10: flow_3_5_2_10 - 1000 edge_3_5_2_10 <= 0
 inactive_no_flow_3_6_2_1: flow_3_6_2_1 - 1000 edge_3_6_2_1 <= 0
 inactive_no_flow_3_8_2_2: flow_3_8_2_2 - 1000 edge_3_8_2_2 <= 0
 inactive_no_flow_3_9_2_5: flow_3_9_2_5 - 1000 edge_3_9_2_5 <= 0
 inactive_no_flow_4_1_3_4: flow_4_1_3_4 - 1000 edge_4_1_3_4 <= 0
 inactive_no_flow_4_2_3_7: flow_4_2_3_7 - 1000 edge_4_2_3_7 <= 0
 inactive_no_flow_4_3_3_2: flow_4_3_3_2 - 1000 edge_4_3_3_2 <= 0
 inactive_no_flow_4_5_3_10: flow_4_5_3_10 - 1000 edge_4_5_3_10 <= 0
 inactive_no_flow_4_6_3_1: flow_4_6_3_1 - 1000 edge_4_6_3_1 <= 0
 inactive_no_flow_4_7_3_7: flow_4_7_3_7 - 1000 edge_4_7_3_7 <= 0
 inactive_no_flow_4_8_3_2: flow_4_8_3_2 - 1000 edge_4_8_3_2 <= 0
 inactive_no_flow_4_9_3_5: flow_4_9_3_5 - 1000 edge_4_9_3_5 <= 0
 inactive_no_flow_5_1_10_4: flow_5_1_10_4 - 1000 edge_5_1_10_4 <= 0
 inactive_no_flow_5_3_10_2: flow_5_3_10_2 - 1000 edge_5_3_10_2 <= 0
 inactive_no_flow_5_4_10_3: flow_5_4_10_3 - 1000 edge_5_4_10_3 <= 0
 inactive_no_flow_5_6_10_1: flow_5_6_10_1 - 1000 edge_5_6_10_1 <= 0
 inactive_no_flow_5_7_10_7: flow_5_7_10_7 - 1000 edge_5_7_10_7 <= 0
 inactive_no_flow_5_8_10_2: flow_5_8_10_2 - 1000 edge_5_8_10_2 <= 0
 inactive_no_flow_5_9_10_5: flow_5_9_10_5 - 1000 edge_5_9_10_5 <= 0
 inactive_no_flow_6_1_1_4: flow_6_1_1_4 - 1000 edge_6_1_1_4 <= 0
 inactive_no_flow_6_3_1_2: flow_6_3_1_2 - 1000 edge_6_3_1_2 <= 0
 inactive_no_flow_6_5_1_10: flow_6_5_1_10 - 1000 edge_6_5_1_10 <= 0
 inactive_no_flow_6_7_1_7: flow_6_7_1_7 - 1000 edge_6_7_1_7 <= 0
 inactive_no_flow_6_8_1_2: flow_6_8_1_2 - 1000 edge_6_8_1_2 <= 0
 inactive_no_flow_6_9_1_5: flow_6_9_1_5 - 1000 edge_6_9_1_5 <= 0
 inactive_no_flow_7_5_7_10: flow_7_5_7_10 - 1000 edge_7_5_7_10 <= 0
 inactive_no_flow_7_6_7_1: flow_7_6_7_1 - 1000 edge_7_6_7_1 <= 0
 inactive_no_flow_8_1_2_4: flow_8_1_2_4 - 1000 edge_8_1_2_4 <= 0
 inactive_no_flow_8_3_2_2: flow_8_3_2_2 - 1000 edge_8_3_2_2 <= 0
 inactive_no_flow_8_4_2_3: flow_8_4_2_3 - 1000 edge_8_4_2_3 <= 0
 inactive_no_flow_8_5_2_10: flow_8_5_2_10 - 1000 edge_8_5_2_10 <= 0
 inactive_no_flow_8_6_2_1: flow_8_6_2_1 - 1000 edge_8_6_2_1 <= 0
 inactive_no_flow_8_7_2_7: flow_8_7_2_7 - 1000 edge_8_7_2_7 <= 0
 inactive_no_flow_8_9_2_5: flow_8_9_2_5 - 1000 edge_8_9_2_5 <= 0
 inactive_no_flow_9_1_5_4: flow_9_1_5_4 - 1000 edge_9_1_5_4 <= 0
 inactive_no_flow_9_5_5_10: flow_9_5_5_10 - 1000 edge_9_5_5_10 <= 0
 inactive_no_flow_9_6_5_1: flow_9_6_5_1 - 1000 edge_9_6_5_1 <= 0
 inactive_no_flow_9_8_5_2: flow_9_8_5_2 - 1000 edge_9_8_5_2 <= 0
 arc_0_1_0_4_pos_must_match_v: - edge_0_1_0_4 + pos_1_4 >= 0
 arc_0_3_0_2_pos_must_match_v: - edge_0_3_0_2 + pos_3_2 >= 0
 arc_0_4_0_3_pos_must_match_v: - edge_0_4_0_3 + pos_4_3 >= 0
 arc_0_5_0_10_pos_must_match_v: - edge_0_5_0_10 + pos_5_10 >= 0
 arc_0_8_0_2_pos_must_match_v: - edge_0_8_0_2 + pos_8_2 >= 0
 arc_1_3_4_2_pos_must_match_u: - edge_1_3_4_2 + pos_1_4 >= 0
 arc_1_3_4_2_pos_must_match_v: - edge_1_3_4_2 + pos_3_2 >= 0
 arc_1_4_4_3_pos_must_match_u: - edge_1_4_4_3 + pos_1_4 >= 0
 arc_1_4_4_3_pos_must_match_v: - edge_1_4_4_3 + pos_4_3 >= 0
 arc_1_5_4_10_pos_must_match_u: - edge_1_5_4_10 + pos_1_4 >= 0
 arc_1_5_4_10_pos_must_match_v: - edge_1_5_4_10 + pos_5_10 >= 0
 arc_1_6_4_1_pos_must_match_u: - edge_1_6_4_1 + pos_1_4 >= 0
 arc_1_6_4_1_pos_must_match_v: - edge_1_6_4_1 + pos_6_1 >= 0
 arc_1_8_4_2_pos_must_match_u: - edge_1_8_4_2 + pos_1_4 >= 0
 arc_1_8_4_2_pos_must_match_v: - edge_1_8_4_2 + pos_8_2 >= 0
 arc_1_9_4_5_pos_must_match_u: - edge_1_9_4_5 + pos_1_4 >= 0
 arc_1_9_4_5_pos_must_match_v: - edge_1_9_4_5 + pos_9_5 >= 0
 arc_2_4_7_3_pos_must_match_u: - edge_2_4_7_3 + pos_2_7 >= 0
 arc_2_4_7_3_pos_must_match_v: - edge_2_4_7_3 + pos_4_3 >= 0
 arc_2_5_7_10_pos_must_match_u: - edge_2_5_7_10 + pos_2_7 >= 0
 arc_2_5_7_10_pos_must_match_v: - edge_2_5_7_10 + pos_5_10 >= 0
 arc_2_8_7_2_pos_must_match_u: - edge_2_8_7_2 + pos_2_7 >= 0
 arc_2_8_7_2_pos_must_match_v: - edge_2_8_7_2 + pos_8_2 >= 0
 arc_3_1_2_4_pos_must_match_u: - edge_3_1_2_4 + pos_3_2 >= 0
 arc_3_1_2_4_pos_must_match_v: - edge_3_1_2_4 + pos_1_4 >= 0
 arc_3_2_2_7_pos_must_match_u: - edge_3_2_2_7 + pos_3_2 >= 0
 arc_3_2_2_7_pos_must_match_v: - edge_3_2_2_7 + pos_2_7 >= 0
 arc_3_4_2_3_pos_must_match_u: - edge_3_4_2_3 + pos_3_2 >= 0
 arc_3_4_2_3_pos_must_match_v: - edge_3_4_2_3 + pos_4_3 >= 0
 arc_3_5_2_10_pos_must_match_u: - edge_3_5_2_10 + pos_3_2 >= 0
 arc_3_5_2_10_pos_must_match_v: - edge_3_5_2_10 + pos_5_10 >= 0
 arc_3_6_2_1_pos_must_match_u: - edge_3_6_2_1 + pos_3_2 >= 0
 arc_3_6_2_1_pos_must_match_v: - edge_3_6_2_1 + pos_6_1 >= 0
 arc_3_8_2_2_pos_must_match_u: - edge_3_8_2_2 + pos_3_2 >= 0
 arc_3_8_2_2_pos_must_match_v: - edge_3_8_2_2 + pos_8_2 >= 0
 arc_3_9_2_5_pos_must_match_u: - edge_3_9_2_5 + pos_3_2 >= 0
 arc_3_9_2_5_pos_must_match_v: - edge_3_9_2_5 + pos_9_5 >= 0
 arc_4_1_3_4_pos_must_match_u: - edge_4_1_3_4 + pos_4_3 >= 0
 arc_4_1_3_4_pos_must_match_v: - edge_4_1_3_4 + pos_1_4 >= 0
 arc_4_2_3_7_pos_must_match_u: - edge_4_2_3_7 + pos_4_3 >= 0
 arc_4_2_3_7_pos_must_match_v: - edge_4_2_3_7 + pos_2_7 >= 0
 arc_4_3_3_2_pos_must_match_u: - edge_4_3_3_2 + pos_4_3 >= 0
 arc_4_3_3_2_pos_must_match_v: - edge_4_3_3_2 + pos_3_2 >= 0
 arc_4_5_3_10_pos_must_match_u: - edge_4_5_3_10 + pos_4_3 >= 0
 arc_4_5_3_10_pos_must_match_v: - edge_4_5_3_10 + pos_5_10 >= 0
 arc_4_6_3_1_pos_must_match_u: - edge_4_6_3_1 + pos_4_3 >= 0
 arc_4_6_3_1_pos_must_match_v: - edge_4_6_3_1 + pos_6_1 >= 0
 arc_4_7_3_7_pos_must_match_u: - edge_4_7_3_7 + pos_4_3 >= 0
 arc_4_7_3_7_pos_must_match_v: - edge_4_7_3_7 + pos_7_7 >= 0
 arc_4_8_3_2_pos_must_match_u: - edge_4_8_3_2 + pos_4_3 >= 0
 arc_4_8_3_2_pos_must_match_v: - edge_4_8_3_2 + pos_8_2 >= 0
 arc_4_9_3_5_pos_must_match_u: - edge_4_9_3_5 + pos_4_3 >= 0
 arc_4_9_3_5_pos_must_match_v: - edge_4_9_3_5 + pos_9_5 >= 0
 arc_5_1_10_4_pos_must_match_u: - edge_5_1_10_4 + pos_5_10 >= 0
 arc_5_1_10_4_pos_must_match_v: - edge_5_1_10_4 + pos_1_4 >= 0
 arc_5_3_10_2_pos_must_match_u: - edge_5_3_10_2 + pos_5_10 >= 0
 arc_5_3_10_2_pos_must_match_v: - edge_5_3_10_2 + pos_3_2 >= 0
 arc_5_4_10_3_pos_must_match_u: - edge_5_4_10_3 + pos_5_10 >= 0
 arc_5_4_10_3_pos_must_match_v: - edge_5_4_10_3 + pos_4_3 >= 0
 arc_5_6_10_1_pos_must_match_u: - edge_5_6_10_1 + pos_5_10 >= 0
 arc_5_6_10_1_pos_must_match_v: - edge_5_6_10_1 + pos_6_1 >= 0
 arc_5_7_10_7_pos_must_match_u: - edge_5_7_10_7 + pos_5_10 >= 0
 arc_5_7_10_7_pos_must_match_v: - edge_5_7_10_7 + pos_7_7 >= 0
 arc_5_8_10_2_pos_must_match_u: - edge_5_8_10_2 + pos_5_10 >= 0
 arc_5_8_10_2_pos_must_match_v: - edge_5_8_10_2 + pos_8_2 >= 0
 arc_5_9_10_5_pos_must_match_u: - edge_5_9_10_5 + pos_5_10 >= 0
 arc_5_9_10_5_pos_must_match_v: - edge_5_9_10_5 + pos_9_5 >= 0
 arc_6_1_1_4_pos_must_match_u: - edge_6_1_1_4 + pos_6_1 >= 0
 arc_6_1_1_4_pos_must_match_v: - edge_6_1_1_4 + pos_1_4 >= 0
 arc_6_3_1_2_pos_must_match_u: - edge_6_3_1_2 + pos_6_1 >= 0
 arc_6_3_1_2_pos_must_match_v: - edge_6_3_1_2 + pos_3_2 >= 0
 arc_6_5_1_10_pos_must_match_u: - edge_6_5_1_10 + pos_6_1 >= 0
 arc_6_5_1_10_pos_must_match_v: - edge_6_5_1_10 + pos_5_10 >= 0
 arc_6_7_1_7_pos_must_match_u: - edge_6_7_1_7 + pos_6_1 >= 0
 arc_6_7_1_7_pos_must_match_v: - edge_6_7_1_7 + pos_7_7 >= 0
 arc_6_8_1_2_pos_must_match_u: - edge_6_8_1_2 + pos_6_1 >= 0
 arc_6_8_1_2_pos_must_match_v: - edge_6_8_1_2 + pos_8_2 >= 0
 arc_6_9_1_5_pos_must_match_u: - edge_6_9_1_5 + pos_6_1 >= 0
 arc_6_9_1_5_pos_must_match_v: - edge_6_9_1_5 + pos_9_5 >= 0
 arc_7_5_7_10_pos_must_match_u: - edge_7_5_7_10 + pos_7_7 >= 0
 arc_7_5_7_10_pos_must_match_v: - edge_7_5_7_10 + pos_5_10 >= 0
 arc_7_6_7_1_pos_must_match_u: - edge_7_6_7_1 + pos_7_7 >= 0
 arc_7_6_7_1_pos_must_match_v: - edge_7_6_7_1 + pos_6_1 >= 0
 arc_8_1_2_4_pos_must_match_u: - edge_8_1_2_4 + pos_8_2 >= 0
 arc_8_1_2_4_pos_must_match_v: - edge_8_1_2_4 + pos_1_4 >= 0
 arc_8_3_2_2_pos_must_match_u: - edge_8_3_2_2 + pos_8_2 >= 0
 arc_8_3_2_2_pos_must_match_v: - edge_8_3_2_2 + pos_3_2 >= 0
 arc_8_4_2_3_pos_must_match_u: - edge_8_4_2_3 + pos_8_2 >= 0
 arc_8_4_2_3_pos_must_match_v: - edge_8_4_2_3 + pos_4_3 >= 0
 arc_8_5_2_10_pos_must_match_u: - edge_8_5_2_10 + pos_8_2 >= 0
 arc_8_5_2_10_pos_must_match_v: - edge_8_5_2_10 + pos_5_10 >= 0
 arc_8_6_2_1_pos_must_match_u: - edge_8_6_2_1 + pos_8_2 >= 0
 arc_8_6_2_1_pos_must_match_v: - edge_8_6_2_1 + pos_6_1 >= 0
 arc_8_7_2_7_pos_must_match_u: - edge_8_7_2_7 + pos_8_2 >= 0
 arc_8_7_2_7_pos_must_match_v: - edge_8_7_2_7 + pos_7_7 >= 0
 arc_8_9_2_5_pos_must_match_u: - edge_8_9_2_5 + pos_8_2 >= 0
 arc_8_9_2_5_pos_must_match_v: - edge_8_9_2_5 + pos_9_5 >= 0
 arc_9_1_5_4_pos_must_match_u: - edge_9_1_5_4 + pos_9_5 >= 0
 arc_9_1_5_4_pos_must_match_v: - edge_9_1_5_4 + pos_1_4 >= 0
 arc_9_5_5_10_pos_must_match_u: - edge_9_5_5_10 + pos_9_5 >= 0
 arc_9_5_5_10_pos_must_match_v: - edge_9_5_5_10 + pos_5_10 >= 0
 arc_9_6_5_1_pos_must_match_u: - edge_9_6_5_1 + pos_9_5 >= 0
 arc_9_6_5_1_pos_must_match_v: - edge_9_6_5_1 + pos_6_1 >= 0
 arc_9_8_5_2_pos_must_match_u: - edge_9_8_5_2 + pos_9_5 >= 0
 arc_9_8_5_2_pos_must_match_v: - edge_9_8_5_2 + pos_8_2 >= 0
Bounds
Binaries
 edge_0_1_0_4 edge_0_3_0_2 edge_0_4_0_3 edge_0_5_0_10 edge_0_8_0_2
 edge_1_3_4_2 edge_1_4_4_3 edge_1_5_4_10 edge_1_6_4_1 edge_1_8_4_2
 edge_1_9_4_5 edge_2_4_7_3 edge_2_5_7_10 edge_2_8_7_2 edge_3_1_2_4
 edge_3_2_2_7 edge_3_4_2_3 edge_3_5_2_10 edge_3_6_2_1 edge_3_8_2_2
 edge_3_9_2_5 edge_4_1_3_4 edge_4_2_3_7 edge_4_3_3_2 edge_4_5_3_10
 edge_4_6_3_1 edge_4_7_3_7 edge_4_8_3_2 edge_4_9_3_5 edge_5_1_10_4
 edge_5_3_10_2 edge_5_4_10_3 edge_5_6_10_1 edge_5_7_10_7 edge_5_8_10_2
 edge_5_9_10_5 edge_6_1_1_4 edge_6_3_1_2 edge_6_5_1_10 edge_6_7_1_7
 edge_6_8_1_2 edge_6_9_1_5 edge_7_5_7_10 edge_7_6_7_1 edge_8_1_2_4
 edge_8_3_2_2 edge_8_4_2_3 edge_8_5_2_10 edge_8_6_2_1 edge_8_7_2_7
 edge_8_9_2_5 edge_9_1_5_4 edge_9_5_5_10 edge_9_6_5_1 edge_9_8_5_2 pos_1_4
 pos_2_7 pos_3_2 pos_4_3 pos_5_10 pos_6_1 pos_7_7 pos_8_2 pos_9_5
End
