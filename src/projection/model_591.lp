Maximize
  - 2.7756206698032 edge_0_2_0_0 - 3.137648190362401 edge_0_3_0_0
   - 2.42403 edge_0_4_0_0 - 2.2625411302496 edge_0_6_0_0
   - 3.829597456399 edge_0_7_0_0 - 2.8149648290704 edge_0_9_0_0
   + 0.4009103733238856 edge_2_3_0_0 + 0.4394072836392 edge_2_4_0_0
   - 0.3502652298926598 edge_2_5_0_0 + 0.1143901156998785 edge_2_6_0_0
   + 0.241644674828579 edge_2_7_0_0 - 0.520150275137869 edge_2_8_0_0
   - 0.5233771865092357 edge_2_9_0_0 - 0.6685906609750617 edge_2_10_0_0
   - 0.8315470900428398 edge_2_12_0_0 + 0.2417538622176039 edge_2_13_0_0
   - 0.3991390608927417 edge_3_2_0_0 - 0.10543598211432 edge_3_4_0_0
   - 0.846503574975692 edge_3_5_0_0 - 0.4513708139375752 edge_3_8_0_0
   + 0.69304630958094 edge_4_2_0_0 + 1.0492096889208 edge_4_3_0_0
   - 0.1049494800819 edge_4_5_0_0 + 0.72380496341626 edge_4_6_0_0
   + 0.17427570466438 edge_4_7_0_0 + 0.084151368606 edge_4_8_0_0
   - 0.30228265402604 edge_4_9_0_0 - 0.4546798363152 edge_4_12_0_0
   + 0.96608147831572 edge_4_13_0_0 - 0.0814214021063439 edge_5_2_0_0
   + 0.285946595821914 edge_5_3_0_0 - 0.5813481031695 edge_5_4_0_0
   + 1.659168967964401 edge_5_6_0_0 + 0.4449081584513099 edge_5_7_0_0
   + 0.1235167809975821 edge_5_8_0_0 + 0.2178964884396396 edge_5_9_0_0
   - 0.4926115909166779 edge_5_10_0_0 - 0.6515775635617926 edge_5_12_0_0
   + 0.2908504418076249 edge_6_2_0_0 + 0.6394394753490986 edge_6_3_0_0
   - 0.70697210380348 edge_6_4_0_0 + 2.289611165060626 edge_6_5_0_0
   + 0.827813146453269 edge_6_7_0_0 + 1.149026384811029 edge_6_8_0_0
   + 2.02459449725447 edge_6_9_0_0 - 0.682931166890256 edge_6_10_0_0
   - 0.0358282882912876 edge_6_12_0_0 + 2.173630697997789 edge_6_13_0_0
   - 1.161260909533594 edge_7_2_0_0 - 0.9017205849601013 edge_7_3_0_0
   - 1.2616157684825 edge_7_4_0_0 - 0.2721554326263738 edge_7_5_0_0
   - 0.5912476028590772 edge_7_6_0_0 + 0.7240063957774184 edge_7_8_0_0
   + 0.2391320759719216 edge_7_9_0_0 - 0.4022933644672157 edge_7_10_0_0
   + 0.5358944660201921 edge_7_13_0_0 - 1.189526910939892 edge_8_5_0_0
   - 1.787821891795168 edge_8_6_0_0 - 0.6460786480330279 edge_8_7_0_0
   - 0.2616118443032842 edge_8_9_0_0 - 1.550315176832621 edge_9_2_0_0
   - 1.23366629449986 edge_9_3_0_0 - 0.1783306059679787 edge_9_5_0_0
   - 1.206766062715199 edge_9_6_0_0 - 1.073300208109898 edge_9_7_0_0
   - 0.1252749202141433 edge_9_8_0_0 - 0.399793380419385 edge_9_10_0_0
   + 0.1117395189153987 edge_9_12_0_0 + 0.9031710999528025 edge_9_13_0_0
   + 1.201557084866094 edge_10_12_0_0 - 0.095519302161469 edge_10_13_0_0
   - 0.6380561800151725 edge_12_2_0_0 - 0.1523111132948995 edge_12_3_0_0
   - 0.3506588052189846 edge_12_5_0_0 - 1.203109532859612 edge_12_6_0_0
   - 0.6346952940037119 edge_12_7_0_0 - 0.4212424106836576 edge_12_8_0_0
   - 0.8253246849876363 edge_12_9_0_0 + 2.77726797200827 edge_12_10_0_0
   + 2.434107809301473 edge_12_13_0_0 - 0.9013048705020471 edge_13_10_0_0
   - 1.23678888688924 edge_13_12_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_9_2_0_0 + edge_12_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 - flow_2_12_0_0 - flow_2_13_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0
   + flow_7_2_0_0 + flow_9_2_0_0 + flow_12_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_9_3_0_0 + edge_12_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_8_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0
   + flow_12_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_12_0_0
   - flow_4_13_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 + edge_12_5_0_0 = 1
 consume_one_flow_unit_5: flow_2_5_0_0 + flow_3_5_0_0 + flow_4_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0
   - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0 - flow_5_10_0_0
   - flow_5_12_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0
   + flow_9_5_0_0 + flow_12_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 + edge_12_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0
   - flow_6_10_0_0 - flow_6_12_0_0 - flow_6_13_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 + flow_9_6_0_0 + flow_12_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_4_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 + edge_12_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_4_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0 - flow_7_3_0_0
   - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0
   - flow_7_9_0_0 - flow_7_10_0_0 - flow_7_13_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 + flow_12_7_0_0 = 1
 one_parent_8: edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 + edge_12_8_0_0 = 1
 consume_one_flow_unit_8: flow_2_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 + flow_9_8_0_0
   + flow_12_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_4_9_0_0 + edge_5_9_0_0
   + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 + edge_12_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_4_9_0_0
   + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_5_0_0 - flow_9_6_0_0
   - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_10_0_0 - flow_9_12_0_0
   - flow_9_13_0_0 + flow_12_9_0_0 = 1
 one_parent_10: edge_2_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0
   + edge_7_10_0_0 + edge_9_10_0_0 + edge_12_10_0_0 + edge_13_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0
   + flow_7_10_0_0 + flow_9_10_0_0 - flow_10_12_0_0 - flow_10_13_0_0
   + flow_12_10_0_0 + flow_13_10_0_0 = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: edge_2_12_0_0 + edge_4_12_0_0 + edge_5_12_0_0
   + edge_6_12_0_0 + edge_9_12_0_0 + edge_10_12_0_0 + edge_13_12_0_0 = 1
 consume_one_flow_unit_12: flow_2_12_0_0 + flow_4_12_0_0 + flow_5_12_0_0
   + flow_6_12_0_0 + flow_9_12_0_0 + flow_10_12_0_0 - flow_12_2_0_0
   - flow_12_3_0_0 - flow_12_5_0_0 - flow_12_6_0_0 - flow_12_7_0_0
   - flow_12_8_0_0 - flow_12_9_0_0 - flow_12_10_0_0 - flow_12_13_0_0
   + flow_13_12_0_0 = 1
 one_parent_13: edge_2_13_0_0 + edge_4_13_0_0 + edge_6_13_0_0
   + edge_7_13_0_0 + edge_9_13_0_0 + edge_10_13_0_0 + edge_12_13_0_0 = 1
 consume_one_flow_unit_13: flow_2_13_0_0 + flow_4_13_0_0 + flow_6_13_0_0
   + flow_7_13_0_0 + flow_9_13_0_0 + flow_10_13_0_0 + flow_12_13_0_0
   - flow_13_10_0_0 - flow_13_12_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_9_0_0 = 13
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_4_13_0_0: flow_4_13_0_0 - 1000 edge_4_13_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_6_13_0_0: flow_6_13_0_0 - 1000 edge_6_13_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_13_0_0: flow_7_13_0_0 - 1000 edge_7_13_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_13_0_0: flow_9_13_0_0 - 1000 edge_9_13_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_10_13_0_0: flow_10_13_0_0 - 1000 edge_10_13_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_5_0_0: flow_12_5_0_0 - 1000 edge_12_5_0_0 <= 0
 inactive_no_flow_12_6_0_0: flow_12_6_0_0 - 1000 edge_12_6_0_0 <= 0
 inactive_no_flow_12_7_0_0: flow_12_7_0_0 - 1000 edge_12_7_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_13_0_0: flow_12_13_0_0 - 1000 edge_12_13_0_0 <= 0
 inactive_no_flow_13_10_0_0: flow_13_10_0_0 - 1000 edge_13_10_0_0 <= 0
 inactive_no_flow_13_12_0_0: flow_13_12_0_0 - 1000 edge_13_12_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_9_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0 edge_2_12_0_0
 edge_2_13_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_8_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0
 edge_4_8_0_0 edge_4_9_0_0 edge_4_12_0_0 edge_4_13_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0
 edge_5_9_0_0 edge_5_10_0_0 edge_5_12_0_0 edge_6_2_0_0 edge_6_3_0_0
 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0
 edge_6_10_0_0 edge_6_12_0_0 edge_6_13_0_0 edge_7_2_0_0 edge_7_3_0_0
 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0
 edge_7_10_0_0 edge_7_13_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0
 edge_8_9_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_9_12_0_0 edge_9_13_0_0
 edge_10_12_0_0 edge_10_13_0_0 edge_12_2_0_0 edge_12_3_0_0 edge_12_5_0_0
 edge_12_6_0_0 edge_12_7_0_0 edge_12_8_0_0 edge_12_9_0_0 edge_12_10_0_0
 edge_12_13_0_0 edge_13_10_0_0 edge_13_12_0_0
End
