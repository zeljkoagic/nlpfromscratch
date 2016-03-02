Maximize
  - 1.5209316103752 edge_0_1_0_0 - 4.1470507765437 edge_0_2_0_0
   - 0.5363944173392999 edge_0_3_0_0 - 1.7917999667964 edge_0_4_0_0
   - 1.6737744778998 edge_0_5_0_0 - 3.3795397622536 edge_0_7_0_0
   + 0.1785363430353899 edge_1_3_0_0 - 0.0488160757465661 edge_1_4_0_0
   + 0.2976439612411679 edge_1_7_0_0 - 0.1212508660408062 edge_2_3_0_0
   - 0.2158305239055782 edge_2_4_0_0 - 0.786241585940734 edge_2_5_0_0
   - 0.8471210731803813 edge_2_7_0_0 + 0.0863829033177493 edge_3_1_0_0
   + 2.146517724293884 edge_3_2_0_0 + 1.871820735162868 edge_3_4_0_0
   + 0.4076395773177804 edge_3_5_0_0 - 0.1291254341343947 edge_3_6_0_0
   + 1.422449103052082 edge_3_7_0_0 - 0.3646602117784271 edge_4_1_0_0
   - 0.8956357162971326 edge_4_2_0_0 + 0.1124638767620876 edge_4_3_0_0
   - 0.3037427828409266 edge_4_5_0_0 + 0.3205406559894313 edge_4_7_0_0
   + 0.7047744837526058 edge_5_2_0_0 + 0.7484565331013679 edge_5_3_0_0
   + 0.6636834352907088 edge_5_4_0_0 + 2.919147507080985 edge_5_6_0_0
   + 3.412792195247678 edge_5_7_0_0 - 0.3379050995421881 edge_6_2_0_0
   - 0.0977900732482007 edge_6_3_0_0 + 0.1247628069747241 edge_6_4_0_0
   + 0.5810970001691595 edge_6_5_0_0 + 0.5722141909665999 edge_6_7_0_0
   - 1.546023503899574 edge_7_1_0_0 - 1.835708870807782 edge_7_3_0_0
   - 1.308644704501017 edge_7_4_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0 + edge_7_1_0_0
   = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_4_0_0
   - flow_1_7_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_7_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_7_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_7_0_0 + flow_5_4_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 + flow_6_5_0_0 = 1
 one_parent_6: edge_3_6_0_0 + edge_5_6_0_0 = 1
 consume_one_flow_unit_6: flow_3_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_1_0_0 - flow_7_3_0_0 - flow_7_4_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_7_0_0 = 7
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_7_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_7_0_0 edge_2_3_0_0
 edge_2_4_0_0 edge_2_5_0_0 edge_2_7_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_7_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_6_2_0_0
 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_7_1_0_0
 edge_7_3_0_0 edge_7_4_0_0
End
