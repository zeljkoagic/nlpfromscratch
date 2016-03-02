Maximize
  - 1.8071804256672 edge_0_2_0_0 - 1.7556546710999 edge_0_4_0_0
   - 1.0484969315491 edge_0_5_0_0 - 1.3276828346952 edge_0_6_0_0
   - 3.5017324613749 edge_0_7_0_0 + 0.0666293982296644 edge_2_4_0_0
   - 0.1385892688378952 edge_2_5_0_0 - 0.1437058634215049 edge_2_6_0_0
   + 0.4142354366159356 edge_2_7_0_0 - 0.3377888998663077 edge_4_2_0_0
   - 0.2930599723988584 edge_4_5_0_0 - 0.2513776344369354 edge_4_6_0_0
   - 0.0289110338133354 edge_4_7_0_0 + 0.0498036655336235 edge_5_2_0_0
   + 0.2135777599222053 edge_5_4_0_0 + 0.4874698426490199 edge_5_6_0_0
   + 0.5908184529584496 edge_5_7_0_0 - 0.0926574051330985 edge_6_2_0_0
   - 0.0384260851011382 edge_6_4_0_0 + 0.178410624431748 edge_6_5_0_0
   + 0.0108867716604578 edge_6_7_0_0 - 0.3285002680147083 edge_7_2_0_0
   - 0.5624496674665006 edge_7_4_0_0 - 0.9688852362659691 edge_7_5_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_4_0_0 - flow_2_5_0_0
   - flow_2_6_0_0 - flow_2_7_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_5_4_0_0 + edge_6_4_0_0
   + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 - flow_4_2_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 + flow_5_4_0_0
   + flow_6_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_4_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_4_5_0_0
   - flow_5_2_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_4_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_4_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_4_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 = 7
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_4_2_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_5_2_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_7_0_0 edge_6_2_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_7_2_0_0 edge_7_4_0_0 edge_7_5_0_0
End
