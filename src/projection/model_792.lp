Maximize
  - 3.6812269947278 edge_0_1_0_0 - 3.00629 edge_0_2_0_0
   - 2.0046266522679 edge_0_3_0_0 - 2.7578162974518 edge_0_4_0_0
   - 3.825670613183 edge_0_5_0_0 - 2.553649080686 edge_0_6_0_0
   - 2.970487613015 edge_0_9_0_0 - 0.27654743890053 edge_1_2_0_0
   - 0.2347756641645161 edge_1_3_0_0 + 0.1023772396647239 edge_1_4_0_0
   - 0.7774949379033511 edge_1_5_0_0 - 0.8455004467459696 edge_1_6_0_0
   - 1.395473106216109 edge_1_7_0_0 - 1.701483816282099 edge_1_9_0_0
   - 0.30188476615401 edge_2_1_0_0 + 0.52321812622427 edge_2_3_0_0
   + 0.15822666842562 edge_2_4_0_0 + 0.33689793453363 edge_2_5_0_0
   - 0.4856248251749999 edge_2_6_0_0 - 0.37818197730908 edge_2_7_0_0
   - 0.99827983715572 edge_2_9_0_0 + 0.1963948939467 edge_2_10_0_0
   + 0.9745814196377213 edge_3_1_0_0 + 2.0045866523347 edge_3_2_0_0
   + 1.263386689749805 edge_3_4_0_0 + 2.512946905359147 edge_3_5_0_0
   + 2.291928102647186 edge_3_6_0_0 + 0.6140202062957854 edge_3_7_0_0
   + 0.1692650434433582 edge_3_8_0_0 + 0.5479809947635306 edge_3_9_0_0
   + 1.383952183056218 edge_3_10_0_0 - 0.5232243392335273 edge_4_1_0_0
   - 0.5948468608329001 edge_4_2_0_0 - 0.3137138298982691 edge_4_3_0_0
   - 0.9383018763188663 edge_5_1_0_0 - 1.2463816432349 edge_5_2_0_0
   - 0.283129930900612 edge_5_3_0_0 + 0.2770358011899757 edge_5_6_0_0
   - 0.5646647357293938 edge_5_7_0_0 - 0.8710535260090597 edge_5_9_0_0
   - 0.1191575390860641 edge_5_10_0_0 - 1.12148147289671 edge_6_1_0_0
   - 0.7381996250769 edge_6_2_0_0 - 0.2665999839579912 edge_6_3_0_0
   - 0.4112332286612382 edge_6_4_0_0 + 0.5217708429042889 edge_6_5_0_0
   + 1.025139569441222 edge_6_7_0_0 + 0.3023399391386733 edge_6_8_0_0
   + 0.7804594645037817 edge_6_9_0_0 + 1.075429032113209 edge_6_10_0_0
   - 0.717808939929661 edge_7_1_0_0 - 0.6807289591562601 edge_7_2_0_0
   + 0.1778433706214771 edge_7_3_0_0 + 0.1512344367265232 edge_7_5_0_0
   + 0.7778216733147768 edge_7_6_0_0 + 0.497251187261567 edge_7_8_0_0
   + 0.5171608124006395 edge_7_9_0_0 + 0.9307114415728301 edge_7_10_0_0
   - 1.064088779353517 edge_8_1_0_0 - 0.9412864072730661 edge_8_3_0_0
   - 0.5896831869203609 edge_8_5_0_0 - 0.5247881496945981 edge_8_6_0_0
   + 0.2177655107226286 edge_8_7_0_0 + 0.2712136170117677 edge_8_10_0_0
   - 0.5225924617409572 edge_9_1_0_0 + 0.056766974508531 edge_9_2_0_0
   + 0.5786645433990917 edge_9_3_0_0 - 0.1120405440425012 edge_9_4_0_0
   + 0.0491361167540457 edge_9_5_0_0 + 0.7817599209148025 edge_9_6_0_0
   + 1.862561070144453 edge_9_7_0_0 + 3.532369337917059 edge_9_8_0_0
   + 1.552932685656097 edge_9_10_0_0 - 1.632869020278053 edge_10_7_0_0
   - 1.440453215436644 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0
   + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0
   - flow_1_9_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0
   + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_9_0_0 - flow_2_10_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0
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
   + edge_6_4_0_0 + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   + flow_6_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_9_0_0 - flow_5_10_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_1_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0
   - flow_7_9_0_0 - flow_7_10_0_0 + flow_8_7_0_0 + flow_9_7_0_0
   + flow_10_7_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0
   = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_3_0_0 - flow_8_5_0_0 - flow_8_6_0_0
   - flow_8_7_0_0 - flow_8_10_0_0 + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0
   - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_2_10_0_0 + edge_3_10_0_0 + edge_5_10_0_0
   + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_3_10_0_0 + flow_5_10_0_0
   + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0 + flow_9_10_0_0
   - flow_10_7_0_0 - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_9_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
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
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_9_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0
 edge_1_5_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_9_0_0 edge_2_1_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0
 edge_2_9_0_0 edge_2_10_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0
 edge_3_10_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_5_1_0_0
 edge_5_2_0_0 edge_5_3_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_9_0_0
 edge_5_10_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0
 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0
 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0 edge_8_1_0_0 edge_8_3_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_10_0_0 edge_9_1_0_0
 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_10_7_0_0 edge_10_9_0_0
End
