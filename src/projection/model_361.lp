Maximize
  - 1.0354177290672 edge_0_2_0_0 - 0.9266844097169999 edge_0_5_0_0
   - 2.6312825906238 edge_0_9_0_0 - 2.7646298147916 edge_0_10_0_0
   - 2.56611606798 edge_0_13_0_0 - 1.9418491440167 edge_0_15_0_0
   - 0.7236622682 edge_0_16_0_0 + 0.0358517518462512 edge_1_2_0_0
   - 0.0179794225675789 edge_1_6_0_0 + 0.1361660154402267 edge_2_1_0_0
   - 0.0883121867316063 edge_2_2_0_0 + 0.2033261716009373 edge_2_5_0_0
   + 0.0414183474079675 edge_2_6_0_0 + 0.0732300921566458 edge_2_9_0_0
   + 0.2562641584704841 edge_2_10_0_0 - 0.1102106033091797 edge_2_12_0_0
   - 0.1072063978987249 edge_2_13_0_0 - 0.0261445742104664 edge_2_15_0_0
   - 0.0772367905259611 edge_2_16_0_0 + 0.4611328009175971 edge_2_23_0_0
   + 0.2873865696652962 edge_5_1_0_0 + 0.2650079752867048 edge_5_2_0_0
   + 0.0618192718888746 edge_5_5_0_0 + 0.4850933953056559 edge_5_6_0_0
   + 0.4843993372083716 edge_5_9_0_0 + 0.7201381136318008 edge_5_10_0_0
   - 0.1523017716246561 edge_5_12_0_0 + 1.270598366200606 edge_5_13_0_0
   + 0.1401607120495437 edge_5_15_0_0 + 0.0568709959429938 edge_5_16_0_0
   + 1.380785736895296 edge_5_23_0_0 - 0.0970970672375089 edge_6_1_0_0
   + 0.0562650930681349 edge_9_1_0_0 - 0.2059421156758452 edge_9_2_0_0
   + 0.4384499216355573 edge_9_5_0_0 + 0.0567842256451025 edge_9_6_0_0
   + 0.8024184869821999 edge_9_10_0_0 + 0.0690269523545417 edge_9_13_0_0
   + 1.245049992791922 edge_9_23_0_0 - 0.0112236841377231 edge_10_1_0_0
   + 0.3597935320022663 edge_10_2_0_0 + 0.7837565522689668 edge_10_5_0_0
   + 0.0845015163367245 edge_10_6_0_0 + 2.654975578440264 edge_10_9_0_0
   + 0.3088619548915807 edge_10_13_0_0 - 0.2740640161497223 edge_10_15_0_0
   + 1.608002593119154 edge_10_23_0_0 - 0.018046871979498 edge_12_16_0_0
   + 0.0640992378765848 edge_13_1_0_0 - 0.1679086817046474 edge_13_2_0_0
   + 0.4754018993776727 edge_13_5_0_0 - 0.0259108170914574 edge_13_6_0_0
   + 0.036358765902131 edge_13_9_0_0 - 0.0308752491257663 edge_13_10_0_0
   - 0.0740593995581171 edge_15_2_0_0 + 0.0425949620480926 edge_15_5_0_0
   - 0.1468019582554308 edge_15_6_0_0 - 0.5729911788627335 edge_15_9_0_0
   - 0.5065294971977714 edge_15_10_0_0 + 0.1394237526084646 edge_15_12_0_0
   - 0.2063420514065317 edge_15_13_0_0 + 0.061515673823204 edge_15_16_0_0
   + 0.7401858396399777 edge_15_23_0_0 - 0.0276008153769314 edge_16_2_0_0
   + 0.0605591147151747 edge_16_12_0_0 + 0.1983931166567307 edge_16_15_0_0
   + 0.3156711958960081 edge_16_23_0_0
Subject To
 one_parent_1: edge_2_1_0_0 + edge_5_1_0_0 + edge_6_1_0_0 + edge_9_1_0_0
   + edge_10_1_0_0 + edge_13_1_0_0 = 1
 consume_one_flow_unit_1: - flow_1_2_0_0 - flow_1_6_0_0 + flow_2_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0
   + flow_13_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_2_2_0_0 + edge_5_2_0_0
   + edge_9_2_0_0 + edge_10_2_0_0 + edge_13_2_0_0 + edge_15_2_0_0
   + edge_16_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_9_0_0 - flow_2_10_0_0
   - flow_2_12_0_0 - flow_2_13_0_0 - flow_2_15_0_0 - flow_2_16_0_0
   - flow_2_23_0_0 + flow_5_2_0_0 + flow_9_2_0_0 + flow_10_2_0_0
   + flow_13_2_0_0 + flow_15_2_0_0 + flow_16_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_5_5_0_0 + edge_9_5_0_0
   + edge_10_5_0_0 + edge_13_5_0_0 + edge_15_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 - flow_5_1_0_0
   - flow_5_2_0_0 - flow_5_6_0_0 - flow_5_9_0_0 - flow_5_10_0_0
   - flow_5_12_0_0 - flow_5_13_0_0 - flow_5_15_0_0 - flow_5_16_0_0
   - flow_5_23_0_0 + flow_9_5_0_0 + flow_10_5_0_0 + flow_13_5_0_0
   + flow_15_5_0_0 = 1
 one_parent_6: edge_1_6_0_0 + edge_2_6_0_0 + edge_5_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 + edge_13_6_0_0 + edge_15_6_0_0 = 1
 consume_one_flow_unit_6: flow_1_6_0_0 + flow_2_6_0_0 + flow_5_6_0_0
   - flow_6_1_0_0 + flow_9_6_0_0 + flow_10_6_0_0 + flow_13_6_0_0
   + flow_15_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_5_9_0_0 + edge_10_9_0_0
   + edge_13_9_0_0 + edge_15_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_5_9_0_0
   - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_5_0_0 - flow_9_6_0_0
   - flow_9_10_0_0 - flow_9_13_0_0 - flow_9_23_0_0 + flow_10_9_0_0
   + flow_13_9_0_0 + flow_15_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_5_10_0_0
   + edge_9_10_0_0 + edge_13_10_0_0 + edge_15_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_5_10_0_0
   + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_5_0_0
   - flow_10_6_0_0 - flow_10_9_0_0 - flow_10_13_0_0 - flow_10_15_0_0
   - flow_10_23_0_0 + flow_13_10_0_0 + flow_15_10_0_0 = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: edge_2_12_0_0 + edge_5_12_0_0 + edge_15_12_0_0
   + edge_16_12_0_0 = 1
 consume_one_flow_unit_12: flow_2_12_0_0 + flow_5_12_0_0 - flow_12_16_0_0
   + flow_15_12_0_0 + flow_16_12_0_0 = 1
 one_parent_13: edge_0_13_0_0 + edge_2_13_0_0 + edge_5_13_0_0
   + edge_9_13_0_0 + edge_10_13_0_0 + edge_15_13_0_0 = 1
 consume_one_flow_unit_13: flow_0_13_0_0 + flow_2_13_0_0 + flow_5_13_0_0
   + flow_9_13_0_0 + flow_10_13_0_0 - flow_13_1_0_0 - flow_13_2_0_0
   - flow_13_5_0_0 - flow_13_6_0_0 - flow_13_9_0_0 - flow_13_10_0_0
   + flow_15_13_0_0 = 1
 one_parent_14: = 1
 consume_one_flow_unit_14: = 1
 one_parent_15: edge_0_15_0_0 + edge_2_15_0_0 + edge_5_15_0_0
   + edge_10_15_0_0 + edge_16_15_0_0 = 1
 consume_one_flow_unit_15: flow_0_15_0_0 + flow_2_15_0_0 + flow_5_15_0_0
   + flow_10_15_0_0 - flow_15_2_0_0 - flow_15_5_0_0 - flow_15_6_0_0
   - flow_15_9_0_0 - flow_15_10_0_0 - flow_15_12_0_0 - flow_15_13_0_0
   - flow_15_16_0_0 - flow_15_23_0_0 + flow_16_15_0_0 = 1
 one_parent_16: edge_0_16_0_0 + edge_2_16_0_0 + edge_5_16_0_0
   + edge_12_16_0_0 + edge_15_16_0_0 = 1
 consume_one_flow_unit_16: flow_0_16_0_0 + flow_2_16_0_0 + flow_5_16_0_0
   + flow_12_16_0_0 + flow_15_16_0_0 - flow_16_2_0_0 - flow_16_12_0_0
   - flow_16_15_0_0 - flow_16_23_0_0 = 1
 one_parent_17: = 1
 consume_one_flow_unit_17: = 1
 one_parent_18: = 1
 consume_one_flow_unit_18: = 1
 one_parent_19: = 1
 consume_one_flow_unit_19: = 1
 one_parent_20: = 1
 consume_one_flow_unit_20: = 1
 one_parent_21: = 1
 consume_one_flow_unit_21: = 1
 one_parent_22: = 1
 consume_one_flow_unit_22: = 1
 one_parent_23: edge_2_23_0_0 + edge_5_23_0_0 + edge_9_23_0_0
   + edge_10_23_0_0 + edge_15_23_0_0 + edge_16_23_0_0 = 1
 consume_one_flow_unit_23: flow_2_23_0_0 + flow_5_23_0_0 + flow_9_23_0_0
   + flow_10_23_0_0 + flow_15_23_0_0 + flow_16_23_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_5_0_0 + flow_0_9_0_0 + flow_0_10_0_0
   + flow_0_13_0_0 + flow_0_15_0_0 + flow_0_16_0_0 = 23
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_13_0_0: flow_0_13_0_0 - 1000 edge_0_13_0_0 <= 0
 inactive_no_flow_0_15_0_0: flow_0_15_0_0 - 1000 edge_0_15_0_0 <= 0
 inactive_no_flow_0_16_0_0: flow_0_16_0_0 - 1000 edge_0_16_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_2_0_0: flow_2_2_0_0 - 1000 edge_2_2_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_2_15_0_0: flow_2_15_0_0 - 1000 edge_2_15_0_0 <= 0
 inactive_no_flow_2_16_0_0: flow_2_16_0_0 - 1000 edge_2_16_0_0 <= 0
 inactive_no_flow_2_23_0_0: flow_2_23_0_0 - 1000 edge_2_23_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_5_0_0: flow_5_5_0_0 - 1000 edge_5_5_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_5_13_0_0: flow_5_13_0_0 - 1000 edge_5_13_0_0 <= 0
 inactive_no_flow_5_15_0_0: flow_5_15_0_0 - 1000 edge_5_15_0_0 <= 0
 inactive_no_flow_5_16_0_0: flow_5_16_0_0 - 1000 edge_5_16_0_0 <= 0
 inactive_no_flow_5_23_0_0: flow_5_23_0_0 - 1000 edge_5_23_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_13_0_0: flow_9_13_0_0 - 1000 edge_9_13_0_0 <= 0
 inactive_no_flow_9_23_0_0: flow_9_23_0_0 - 1000 edge_9_23_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_13_0_0: flow_10_13_0_0 - 1000 edge_10_13_0_0 <= 0
 inactive_no_flow_10_15_0_0: flow_10_15_0_0 - 1000 edge_10_15_0_0 <= 0
 inactive_no_flow_10_23_0_0: flow_10_23_0_0 - 1000 edge_10_23_0_0 <= 0
 inactive_no_flow_12_16_0_0: flow_12_16_0_0 - 1000 edge_12_16_0_0 <= 0
 inactive_no_flow_13_1_0_0: flow_13_1_0_0 - 1000 edge_13_1_0_0 <= 0
 inactive_no_flow_13_2_0_0: flow_13_2_0_0 - 1000 edge_13_2_0_0 <= 0
 inactive_no_flow_13_5_0_0: flow_13_5_0_0 - 1000 edge_13_5_0_0 <= 0
 inactive_no_flow_13_6_0_0: flow_13_6_0_0 - 1000 edge_13_6_0_0 <= 0
 inactive_no_flow_13_9_0_0: flow_13_9_0_0 - 1000 edge_13_9_0_0 <= 0
 inactive_no_flow_13_10_0_0: flow_13_10_0_0 - 1000 edge_13_10_0_0 <= 0
 inactive_no_flow_15_2_0_0: flow_15_2_0_0 - 1000 edge_15_2_0_0 <= 0
 inactive_no_flow_15_5_0_0: flow_15_5_0_0 - 1000 edge_15_5_0_0 <= 0
 inactive_no_flow_15_6_0_0: flow_15_6_0_0 - 1000 edge_15_6_0_0 <= 0
 inactive_no_flow_15_9_0_0: flow_15_9_0_0 - 1000 edge_15_9_0_0 <= 0
 inactive_no_flow_15_10_0_0: flow_15_10_0_0 - 1000 edge_15_10_0_0 <= 0
 inactive_no_flow_15_12_0_0: flow_15_12_0_0 - 1000 edge_15_12_0_0 <= 0
 inactive_no_flow_15_13_0_0: flow_15_13_0_0 - 1000 edge_15_13_0_0 <= 0
 inactive_no_flow_15_16_0_0: flow_15_16_0_0 - 1000 edge_15_16_0_0 <= 0
 inactive_no_flow_15_23_0_0: flow_15_23_0_0 - 1000 edge_15_23_0_0 <= 0
 inactive_no_flow_16_2_0_0: flow_16_2_0_0 - 1000 edge_16_2_0_0 <= 0
 inactive_no_flow_16_12_0_0: flow_16_12_0_0 - 1000 edge_16_12_0_0 <= 0
 inactive_no_flow_16_15_0_0: flow_16_15_0_0 - 1000 edge_16_15_0_0 <= 0
 inactive_no_flow_16_23_0_0: flow_16_23_0_0 - 1000 edge_16_23_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_5_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_0_13_0_0
 edge_0_15_0_0 edge_0_16_0_0 edge_1_2_0_0 edge_1_6_0_0 edge_2_1_0_0
 edge_2_2_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_9_0_0 edge_2_10_0_0
 edge_2_12_0_0 edge_2_13_0_0 edge_2_15_0_0 edge_2_16_0_0 edge_2_23_0_0
 edge_5_1_0_0 edge_5_2_0_0 edge_5_5_0_0 edge_5_6_0_0 edge_5_9_0_0
 edge_5_10_0_0 edge_5_12_0_0 edge_5_13_0_0 edge_5_15_0_0 edge_5_16_0_0
 edge_5_23_0_0 edge_6_1_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_5_0_0
 edge_9_6_0_0 edge_9_10_0_0 edge_9_13_0_0 edge_9_23_0_0 edge_10_1_0_0
 edge_10_2_0_0 edge_10_5_0_0 edge_10_6_0_0 edge_10_9_0_0 edge_10_13_0_0
 edge_10_15_0_0 edge_10_23_0_0 edge_12_16_0_0 edge_13_1_0_0 edge_13_2_0_0
 edge_13_5_0_0 edge_13_6_0_0 edge_13_9_0_0 edge_13_10_0_0 edge_15_2_0_0
 edge_15_5_0_0 edge_15_6_0_0 edge_15_9_0_0 edge_15_10_0_0 edge_15_12_0_0
 edge_15_13_0_0 edge_15_16_0_0 edge_15_23_0_0 edge_16_2_0_0 edge_16_12_0_0
 edge_16_15_0_0 edge_16_23_0_0
End
