Maximize
  - 2.416203987609 edge_0_1_0_0 - 2.5459958815662 edge_0_4_0_0
   - 2.8898087089716 edge_0_5_0_0 - 1.2184092219014 edge_0_6_0_0
   - 2.1176705073903 edge_0_7_0_0 - 4.80885 edge_0_8_0_0
   - 0.1813215543137231 edge_1_2_0_0 - 0.4174422576806903 edge_1_4_0_0
   + 0.0865413239555207 edge_1_6_0_0 + 0.1006297280636851 edge_1_7_0_0
   + 0.00188395 edge_1_8_0_0 - 0.3693409633410495 edge_3_5_0_0
   - 0.0887003689331296 edge_3_7_0_0 - 0.0654752107123958 edge_4_1_0_0
   + 0.2607242470640773 edge_4_6_0_0 - 0.122707376042917 edge_4_7_0_0
   - 0.5293835645673 edge_4_8_0_0 - 0.2555657959118047 edge_5_3_0_0
   + 0.2556223360826268 edge_5_7_0_0 - 0.127129009816055 edge_5_8_0_0
   + 0.6197261183074034 edge_6_1_0_0 + 0.4958844573262784 edge_6_2_0_0
   + 0.3036332263944309 edge_6_3_0_0 - 0.1754587180325901 edge_6_4_0_0
   + 2.902068009497745 edge_6_5_0_0 + 0.7280566055673314 edge_6_7_0_0
   + 0.8971322828332539 edge_6_8_0_0 + 0.8586245669655229 edge_7_1_0_0
   + 0.7170810152139701 edge_7_2_0_0 + 0.685098601922345 edge_7_3_0_0
   + 2.273444095302946 edge_7_4_0_0 + 1.909407218728571 edge_7_5_0_0
   + 1.161727268479924 edge_7_6_0_0 + 0.494771662085117 edge_7_7_0_0
   + 2.4455898532646 edge_7_8_0_0 - 0.8592526337172 edge_8_6_0_0
   - 0.65613696063178 edge_8_7_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_4_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0
   = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_4_0_0
   - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0 + flow_4_1_0_0
   + flow_6_1_0_0 + flow_7_1_0_0 = 1
 one_parent_2: edge_1_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_1_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: - flow_3_5_0_0 - flow_3_7_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0
   = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 - flow_4_1_0_0
   - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0 + edge_7_5_0_0
   = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_3_5_0_0 - flow_5_3_0_0
   - flow_5_7_0_0 - flow_5_8_0_0 + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_4_6_0_0 + edge_7_6_0_0
   + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_4_6_0_0
   - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_7_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0
   - flow_7_6_0_0 - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_4_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_4_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_6_0_0
   - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_4_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_7_0_0: flow_7_7_0_0 - 1000 edge_7_7_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_1_2_0_0 edge_1_4_0_0 edge_1_6_0_0 edge_1_7_0_0
 edge_1_8_0_0 edge_3_5_0_0 edge_3_7_0_0 edge_4_1_0_0 edge_4_6_0_0
 edge_4_7_0_0 edge_4_8_0_0 edge_5_3_0_0 edge_5_7_0_0 edge_5_8_0_0
 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0
 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 edge_7_7_0_0 edge_7_8_0_0
 edge_8_6_0_0 edge_8_7_0_0
End
