Maximize
  - 3.783695576682 edge_0_1_0_0 - 3.0355066626585 edge_0_2_0_0
   - 1.2887352567 edge_0_3_0_0 - 3.7678554216918 edge_0_4_0_0
   - 2.0692254618344 edge_0_5_0_0 - 3.4313810433879 edge_0_6_0_0
   - 3.9224516713928 edge_0_8_0_0 - 1.0476251261245 edge_0_10_0_0
   + 0.1217850826339113 edge_1_2_0_0 + 0.1409846802126928 edge_1_3_0_0
   + 0.3411188387421639 edge_1_4_0_0 - 0.2294822057653837 edge_1_5_0_0
   + 0.0527632305167632 edge_1_6_0_0 - 0.4167768511362393 edge_1_10_0_0
   + 0.1620807856662508 edge_1_11_0_0 - 0.040608291937973 edge_2_1_0_0
   + 1.040386307147785 edge_2_3_0_0 + 0.5376570638748759 edge_2_4_0_0
   - 0.036954566644631 edge_2_5_0_0 + 0.300235112136667 edge_2_6_0_0
   + 0.0870176525778776 edge_2_8_0_0 - 0.3478490163733866 edge_2_10_0_0
   + 1.026170797856026 edge_2_11_0_0 + 1.715688801334372 edge_3_1_0_0
   + 3.140456981625122 edge_3_2_0_0 + 2.575985233118601 edge_3_4_0_0
   + 1.413001669558974 edge_3_5_0_0 + 1.853493046540872 edge_3_6_0_0
   + 0.6585126761847115 edge_3_8_0_0 - 0.1715861199964377 edge_3_9_0_0
   + 2.12342967967906 edge_3_10_0_0 + 2.349281855280713 edge_3_11_0_0
   - 1.348160730646292 edge_4_1_0_0 - 1.137640700242311 edge_4_2_0_0
   - 0.8478372421196027 edge_4_3_0_0 - 0.2919345978333542 edge_4_5_0_0
   + 0.00299608045829588 edge_4_6_0_0 - 0.5503983196820221 edge_4_10_0_0
   - 0.2052552953780006 edge_4_11_0_0 - 0.2926387263644785 edge_5_1_0_0
   - 0.0174188250485776 edge_5_2_0_0 + 0.0160399458635421 edge_5_3_0_0
   + 0.4155751827445095 edge_5_4_0_0 + 0.212993042786363 edge_5_5_0_0
   + 0.405431634403825 edge_5_6_0_0 + 0.4543264902899317 edge_5_8_0_0
   - 0.0492642813282328 edge_5_9_0_0 + 0.2893854822370236 edge_5_10_0_0
   + 0.2451812315322835 edge_5_11_0_0 - 1.75254248317926 edge_6_1_0_0
   - 1.650167030439966 edge_6_2_0_0 - 1.711752230367659 edge_6_3_0_0
   - 1.218243238846439 edge_6_4_0_0 - 0.4130436213749646 edge_6_5_0_0
   - 0.1342758723451458 edge_6_8_0_0 - 0.0174382756552394 edge_6_10_0_0
   - 0.0541780897576767 edge_6_11_0_0 - 0.6917058836981839 edge_8_1_0_0
   - 0.166865627687066 edge_8_2_0_0 - 0.3122566474608727 edge_8_3_0_0
   - 0.232206471334151 edge_8_4_0_0 + 0.1986296781897253 edge_8_5_0_0
   + 0.8934006050704948 edge_8_6_0_0 + 0.5397346990523959 edge_8_9_0_0
   + 0.1627551076143581 edge_8_11_0_0 - 0.857669014686234 edge_9_1_0_0
   - 0.034480868088956 edge_9_5_0_0 + 0.0714400450121219 edge_9_6_0_0
   + 0.9329129589396044 edge_9_8_0_0 - 0.4317853746410647 edge_9_11_0_0
   + 1.015737695617841 edge_10_1_0_0 + 0.9939817803203559 edge_10_2_0_0
   + 0.6062783283175773 edge_10_3_0_0 + 0.4190892202829483 edge_10_4_0_0
   + 1.654378863953788 edge_10_5_0_0 + 1.989201315403786 edge_10_6_0_0
   + 3.252627846185302 edge_10_8_0_0 + 3.536531858230534 edge_10_9_0_0
   + 2.356335704862308 edge_10_11_0_0 - 1.497968989246309 edge_11_8_0_0
   - 1.26529853861392 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0
   + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_10_0_0
   - flow_1_11_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_8_1_0_0 + flow_9_1_0_0
   + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_8_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_8_0_0 - flow_2_10_0_0 - flow_2_11_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_8_2_0_0
   + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_8_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0
   - flow_3_11_0_0 + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_8_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_8_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_10_0_0 - flow_4_11_0_0
   + flow_5_4_0_0 + flow_6_4_0_0 + flow_8_4_0_0 + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_5_5_0_0 + edge_6_5_0_0 + edge_8_5_0_0
   + edge_9_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 - flow_5_10_0_0 - flow_5_11_0_0 + flow_6_5_0_0
   + flow_8_5_0_0 + flow_9_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_8_0_0 - flow_6_10_0_0 - flow_6_11_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0
   - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_6_0_0
   - flow_8_9_0_0 - flow_8_11_0_0 + flow_9_8_0_0 + flow_10_8_0_0
   + flow_11_8_0_0 = 1
 one_parent_9: edge_3_9_0_0 + edge_5_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0
   = 1
 consume_one_flow_unit_9: flow_3_9_0_0 + flow_5_9_0_0 + flow_8_9_0_0
   - flow_9_1_0_0 - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_8_0_0
   - flow_9_11_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0
   - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_8_0_0 - flow_10_9_0_0
   - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_1_11_0_0 + edge_2_11_0_0 + edge_3_11_0_0
   + edge_4_11_0_0 + edge_5_11_0_0 + edge_6_11_0_0 + edge_8_11_0_0
   + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_2_11_0_0 + flow_3_11_0_0
   + flow_4_11_0_0 + flow_5_11_0_0 + flow_6_11_0_0 + flow_8_11_0_0
   + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_8_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_8_0_0 + flow_0_10_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_5_0_0: flow_5_5_0_0 - 1000 edge_5_5_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_8_0_0 edge_0_10_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_1_6_0_0 edge_1_10_0_0 edge_1_11_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_8_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_8_0_0 edge_3_9_0_0
 edge_3_10_0_0 edge_3_11_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_10_0_0 edge_4_11_0_0 edge_5_1_0_0
 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_5_0_0 edge_5_6_0_0
 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_8_0_0
 edge_6_10_0_0 edge_6_11_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_9_0_0 edge_8_11_0_0
 edge_9_1_0_0 edge_9_5_0_0 edge_9_6_0_0 edge_9_8_0_0 edge_9_11_0_0
 edge_10_1_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_5_0_0
 edge_10_6_0_0 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_11_8_0_0
 edge_11_10_0_0
End
