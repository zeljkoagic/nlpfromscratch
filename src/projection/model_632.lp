Maximize
  - 1.944104133881 edge_0_1_0_0 - 3.0865370346426 edge_0_2_0_0
   - 2.5032735506305 edge_0_3_0_0 - 1.1456439157968 edge_0_5_0_0
   - 1.1047147240479 edge_0_7_0_0 - 3.3146754617438 edge_0_8_0_0
   + 1.091072649721127 edge_1_2_0_0 + 0.6374527054441776 edge_1_3_0_0
   + 0.9068631475294625 edge_1_5_0_0 + 0.3920248695442402 edge_1_7_0_0
   + 1.855084364284938 edge_1_8_0_0 + 1.227306710376501 edge_2_1_0_0
   + 0.9479169334832971 edge_2_3_0_0 + 0.4802708237013272 edge_2_5_0_0
   - 0.6786908306985195 edge_2_6_0_0 + 0.5108591026154085 edge_2_7_0_0
   + 0.3382897958512374 edge_2_8_0_0 + 2.293561334271527 edge_3_1_0_0
   + 3.028288192663392 edge_3_2_0_0 + 0.6711564807805983 edge_3_5_0_0
   - 0.1387163038241231 edge_3_6_0_0 + 0.6580269565305414 edge_3_7_0_0
   + 0.7743772753097804 edge_3_8_0_0 + 0.107726185380093 edge_4_1_0_0
   - 0.8953839602627806 edge_4_2_0_0 - 1.142817445495061 edge_4_3_0_0
   - 1.4619548001805 edge_4_8_0_0 + 2.164581815870896 edge_5_1_0_0
   + 1.980798074146033 edge_5_2_0_0 + 3.011517392978959 edge_5_3_0_0
   + 3.583781877016055 edge_5_4_0_0 + 1.115166816258462 edge_5_6_0_0
   + 2.9653562933 edge_5_7_0_0 + 2.75958973970807 edge_5_8_0_0
   - 0.6063003952820899 edge_6_1_0_0 - 0.7609118853108459 edge_6_2_0_0
   - 0.7617827517273109 edge_6_3_0_0 - 0.1837725575450259 edge_6_5_0_0
   - 0.33585850953936 edge_6_8_0_0 + 0.6330289884630838 edge_7_1_0_0
   + 1.084384191054403 edge_7_2_0_0 + 1.858557869573516 edge_7_3_0_0
   + 0.7775982149026001 edge_7_4_0_0 + 2.706386205141539 edge_7_5_0_0
   + 3.316816629211729 edge_7_6_0_0 + 3.115999704062083 edge_7_8_0_0
   - 1.846898359367214 edge_8_1_0_0 - 2.1238698007049 edge_8_3_0_0
   - 0.6777401427313197 edge_8_5_0_0 - 0.6504375316846658 edge_8_7_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_7_0_0 - flow_1_8_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_5_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_8_0_0 + flow_5_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0
   - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: edge_2_6_0_0 + edge_3_6_0_0 + edge_5_6_0_0 + edge_7_6_0_0
   = 1
 consume_one_flow_unit_6: flow_2_6_0_0 + flow_3_6_0_0 + flow_5_6_0_0
   - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_5_0_0
   - flow_6_8_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_5_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_5_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 - flow_8_1_0_0 - flow_8_3_0_0 - flow_8_5_0_0
   - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_5_0_0 edge_1_7_0_0
 edge_1_8_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_8_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_5_0_0 edge_6_8_0_0 edge_7_1_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_8_1_0_0 edge_8_3_0_0 edge_8_5_0_0 edge_8_7_0_0
End
