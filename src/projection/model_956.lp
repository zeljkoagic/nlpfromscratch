Maximize
  - 5.0377539136995 edge_0_1_0_0 - 4.3367904903963 edge_0_2_0_0
   - 1.9094408863054 edge_0_3_0_0 - 3.847982129777 edge_0_4_0_0
   - 3.69656828225 edge_0_5_0_0 - 4.2030309692175 edge_0_6_0_0
   - 3.6249147291801 edge_0_7_0_0 - 4.378699211834 edge_0_8_0_0
   + 0.0144841414752178 edge_1_2_0_0 - 0.43813851866603 edge_1_3_0_0
   - 0.1706032627485951 edge_1_4_0_0 - 0.7965490739953304 edge_1_5_0_0
   - 1.267204583678707 edge_1_7_0_0 - 0.22064491607893 edge_1_8_0_0
   - 0.051560081506296 edge_2_1_0_0 + 0.8517845911432 edge_2_3_0_0
   + 0.2849767818057691 edge_2_4_0_0 - 0.0367130978677046 edge_2_5_0_0
   - 0.2151041854638254 edge_2_6_0_0 + 0.242025079799386 edge_2_7_0_0
   - 0.0499598062643913 edge_2_8_0_0 + 1.2720692187525 edge_3_1_0_0
   + 3.072809762031783 edge_3_2_0_0 + 2.440940683602172 edge_3_4_0_0
   + 0.625382501725 edge_3_5_0_0 + 0.6945941035414245 edge_3_6_0_0
   + 1.2059464746394 edge_3_7_0_0 + 2.149238099044473 edge_3_8_0_0
   - 0.9910396232257143 edge_4_1_0_0 - 0.6470707160466772 edge_4_2_0_0
   - 0.30146110407084 edge_4_3_0_0 + 0.4813093846782012 edge_4_5_0_0
   - 0.5363194317001972 edge_4_6_0_0 + 0.0115726771384683 edge_4_7_0_0
   + 0.37032970110532 edge_4_8_0_0 - 0.5677727446307544 edge_5_1_0_0
   - 0.1114394492854099 edge_5_2_0_0 - 0.1016905844 edge_5_3_0_0
   + 0.6292163804324307 edge_5_4_0_0 + 1.184476083341879 edge_5_6_0_0
   + 1.788893642311882 edge_5_7_0_0 + 2.0746902155 edge_5_8_0_0
   - 0.9458391674677957 edge_6_2_0_0 + 0.2102169506270103 edge_6_3_0_0
   + 0.1236679755628393 edge_6_4_0_0 + 1.314923862457105 edge_6_5_0_0
   + 2.256134434200729 edge_6_8_0_0 - 0.4480529472615999 edge_7_1_0_0
   - 0.0599580584452271 edge_7_2_0_0 + 0.012403776366734 edge_7_3_0_0
   + 0.5878840380652187 edge_7_4_0_0 + 1.18430744610919 edge_7_5_0_0
   + 1.6130601364813 edge_7_8_0_0 - 0.814912945975 edge_8_5_0_0
   - 0.9917140496739389 edge_8_6_0_0 - 0.17484860567474 edge_8_7_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_7_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_7_0_0 - flow_1_8_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0
   + flow_7_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_8_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_8_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_8_0_0 + flow_8_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0
   - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_7_0_0 edge_0_8_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_2_1_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0
 edge_2_8_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0
 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0
 edge_6_5_0_0 edge_6_8_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0
 edge_7_4_0_0 edge_7_5_0_0 edge_7_8_0_0 edge_8_5_0_0 edge_8_6_0_0
 edge_8_7_0_0
End
