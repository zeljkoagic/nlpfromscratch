Maximize
  - 2.889439938931 edge_0_1_0_0 - 2.181058156086 edge_0_2_0_0
   - 2.6084187479584 edge_0_3_0_0 - 2.61199984328 edge_0_5_0_0
   - 2.66733 edge_0_6_0_0 - 3.2294400132774 edge_0_7_0_0
   - 0.1272730404620193 edge_1_2_0_0 - 0.0962040541278839 edge_1_3_0_0
   - 0.0409865928227779 edge_1_5_0_0 - 0.8803262945015 edge_1_6_0_0
   - 0.3854384405828697 edge_1_7_0_0 + 0.0632440483000695 edge_2_1_0_0
   + 0.9017284017369595 edge_2_3_0_0 + 0.8158840389653547 edge_2_5_0_0
   + 2.040299537718476 edge_2_6_0_0 + 0.9284879424870404 edge_2_7_0_0
   + 0.2982648030920845 edge_3_1_0_0 + 1.291263799322981 edge_3_2_0_0
   + 0.8171439459675607 edge_3_5_0_0 + 2.945886638948993 edge_3_6_0_0
   + 0.5933176544747178 edge_3_7_0_0 + 0.50667350169894 edge_5_1_0_0
   + 1.214882189567064 edge_5_2_0_0 + 3.079015884211914 edge_5_3_0_0
   + 1.813771674754854 edge_5_6_0_0 + 1.014369391378033 edge_5_7_0_0
   - 0.045217983978552 edge_6_1_0_0 + 3.265318389443167 edge_6_2_0_0
   + 1.2858793827776 edge_6_3_0_0 + 2.8302598301844 edge_6_5_0_0
   + 1.5711291515898 edge_6_7_0_0 - 1.65111471321866 edge_7_1_0_0
   - 1.905391151956222 edge_7_2_0_0 - 1.844588118518678 edge_7_3_0_0
   - 1.960948823430064 edge_7_5_0_0 - 1.46681719355386 edge_7_6_0_0
   + 0 pos_1_0 + 0 pos_2_0 + 0 pos_3_0 + 0 pos_5_0 + 0 pos_6_0 + 0 pos_7_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   + flow_3_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_5_0_0 - flow_6_7_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_5_0_0 - flow_7_6_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 = 7
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_5_0_0 edge_1_6_0_0
 edge_1_7_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_5_0_0 edge_3_6_0_0
 edge_3_7_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_5_0_0
 edge_7_6_0_0 pos_1_0 pos_2_0 pos_3_0 pos_5_0 pos_6_0 pos_7_0
End
