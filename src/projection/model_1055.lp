Maximize
  - 5.0746078402938 edge_0_1_0_0 - 3.9563546957636 edge_0_2_0_0
   - 1.3444581583575 edge_0_3_0_0 - 3.5744570355624 edge_0_4_0_0
   - 2.1550887862964 edge_0_5_0_0 - 4.1023511158314 edge_0_6_0_0
   - 0.44375667739845 edge_0_7_0_0 - 1.3389824150529 edge_0_8_0_0
   - 1.631181041932 edge_0_9_0_0 - 3.3061232032862 edge_0_10_0_0
   - 0.7254068618660644 edge_1_2_0_0 - 0.2574806892219126 edge_1_3_0_0
   - 0.2104011523204297 edge_1_4_0_0 - 0.1860023284030459 edge_1_5_0_0
   - 0.7092596432935528 edge_1_6_0_0 - 1.166792790333484 edge_1_7_0_0
   - 0.6827856498070591 edge_1_8_0_0 - 0.8709057175710813 edge_1_10_0_0
   - 0.2024480479850322 edge_2_1_0_0 + 0.6681183903693764 edge_2_3_0_0
   - 0.3126991108431413 edge_2_4_0_0 - 0.5264848200140982 edge_2_5_0_0
   - 0.0427024151554301 edge_2_6_0_0 + 0.0825679780235059 edge_2_7_0_0
   - 0.3240678607593799 edge_2_8_0_0 - 1.173111661455639 edge_2_9_0_0
   + 0.0492075987767374 edge_2_10_0_0 + 0.8743118080686894 edge_3_1_0_0
   + 3.509581807789465 edge_3_2_0_0 + 1.214130021179771 edge_3_4_0_0
   + 0.4326755701780756 edge_3_5_0_0 + 1.816737112316878 edge_3_6_0_0
   + 0.4798336656931146 edge_3_7_0_0 + 0.8888227614959222 edge_3_8_0_0
   - 0.3164571943956824 edge_3_9_0_0 + 1.614319304174156 edge_3_10_0_0
   - 1.063499677973143 edge_4_1_0_0 - 0.2340755796611524 edge_4_3_0_0
   + 0.0363190523154201 edge_4_5_0_0 - 0.356645678562136 edge_4_7_0_0
   - 0.3609876109572888 edge_4_8_0_0 - 0.7795025593166374 edge_5_1_0_0
   - 0.2803395057886617 edge_5_3_0_0 + 0.0584027667695846 edge_5_4_0_0
   + 0.1777278005400145 edge_5_7_0_0 - 0.023088972384497 edge_5_8_0_0
   - 1.149867455431427 edge_6_1_0_0 - 0.6204897580680878 edge_6_2_0_0
   - 0.4695431000678702 edge_6_3_0_0 - 0.2837160684707556 edge_6_7_0_0
   - 0.1912529949684724 edge_6_8_0_0 - 0.8558102273063558 edge_6_9_0_0
   - 0.2702443049199595 edge_6_10_0_0 - 0.7316606459307541 edge_7_1_0_0
   + 0.0417184233189518 edge_7_2_0_0 + 0.7705730133400706 edge_7_3_0_0
   - 0.5930856247931822 edge_7_4_0_0 + 0.160748820413851 edge_7_5_0_0
   + 0.2523431331505083 edge_7_6_0_0 + 0.7981942329212013 edge_7_8_0_0
   + 0.3275375588678565 edge_7_9_0_0 + 0.9051932754061939 edge_7_10_0_0
   - 0.0516637751675847 edge_8_1_0_0 + 0.3169784589323103 edge_8_2_0_0
   + 0.0754363264942241 edge_8_3_0_0 - 0.1238455662928759 edge_8_4_0_0
   - 0.0470871163581584 edge_8_5_0_0 + 0.3544008041897926 edge_8_6_0_0
   + 1.352398008816433 edge_8_7_0_0 + 1.306414344167508 edge_8_9_0_0
   + 1.52840150587374 edge_8_10_0_0 + 0.2351793877215944 edge_9_2_0_0
   + 0.0148602351754349 edge_9_3_0_0 + 0.8862037322734798 edge_9_6_0_0
   + 0.2577609033877585 edge_9_7_0_0 + 1.2200533117495 edge_9_8_0_0
   + 1.169601771852721 edge_9_10_0_0 - 1.796755954788459 edge_10_1_0_0
   - 0.897906241428978 edge_10_6_0_0 - 0.8011464748574114 edge_10_7_0_0
   - 0.1772883921482704 edge_10_8_0_0 - 0.5047679168133405 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0
   + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0
   - flow_1_8_0_0 - flow_1_10_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0
   + flow_8_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_10_0_0
   + flow_3_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0
   + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0
   + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   - flow_3_10_0_0 + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_7_3_0_0 + flow_8_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_3_0_0 - flow_4_5_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 + flow_5_4_0_0 + flow_7_4_0_0
   + flow_8_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_3_0_0
   - flow_5_4_0_0 - flow_5_7_0_0 - flow_5_8_0_0 + flow_7_5_0_0
   + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   + flow_7_6_0_0 + flow_8_6_0_0 + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0
   + edge_9_7_0_0 + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0
   - flow_7_10_0_0 + flow_8_7_0_0 + flow_9_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0
   + edge_9_8_0_0 + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0
   - flow_8_9_0_0 - flow_8_10_0_0 + flow_9_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_6_9_0_0
   + edge_7_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_6_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0 - flow_9_2_0_0
   - flow_9_3_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0
   + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0
   + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_6_0_0 - flow_10_7_0_0
   - flow_10_8_0_0 - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0
   + flow_0_9_0_0 + flow_0_10_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_7_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_0_10_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_5_0_0 edge_1_6_0_0
 edge_1_7_0_0 edge_1_8_0_0 edge_1_10_0_0 edge_2_1_0_0 edge_2_3_0_0
 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0
 edge_2_9_0_0 edge_2_10_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0
 edge_3_10_0_0 edge_4_1_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_7_0_0
 edge_4_8_0_0 edge_5_1_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_7_0_0
 edge_5_8_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_7_1_0_0 edge_7_2_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0
 edge_7_9_0_0 edge_7_10_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_6_0_0 edge_9_7_0_0
 edge_9_8_0_0 edge_9_10_0_0 edge_10_1_0_0 edge_10_6_0_0 edge_10_7_0_0
 edge_10_8_0_0 edge_10_9_0_0
End
