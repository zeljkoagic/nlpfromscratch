Maximize
  - 1.4322178869453 edge_0_1_0_0 - 0.7926650925235 edge_0_3_0_0
   - 1.7943186993086 edge_0_6_0_0 - 1.6673766037416 edge_0_8_0_0
   - 0.9749175901204999 edge_0_9_0_0 + 0.0891065130254418 edge_1_2_0_0
   + 0.2091417297938891 edge_1_3_0_0 + 0.1116840293332258 edge_1_5_0_0
   + 0.0305869815771247 edge_1_8_0_0 + 0.0564334063284665 edge_1_9_0_0
   + 0.6583159127803432 edge_1_10_0_0 + 1.318005886792101 edge_2_1_0_0
   + 0.5607299966200061 edge_2_3_0_0 + 0.1826025453784479 edge_2_4_0_0
   - 0.1177888042115381 edge_2_5_0_0 + 0.5588449563823722 edge_2_6_0_0
   - 0.5575098951965441 edge_2_7_0_0 - 0.1305721472311215 edge_2_8_0_0
   + 1.211348363108467 edge_3_1_0_0 + 2.04277168831654 edge_3_2_0_0
   - 0.0615877011850817 edge_3_3_0_0 + 2.365758742680026 edge_3_4_0_0
   + 0.089049984310524 edge_3_5_0_0 + 1.207616953011184 edge_3_6_0_0
   + 0.2129185493821217 edge_3_7_0_0 + 1.800239707979114 edge_3_8_0_0
   + 0.6076481560939763 edge_3_9_0_0 + 3.764521834243571 edge_3_10_0_0
   - 0.0754732886171478 edge_4_2_0_0 + 0.4522990044360024 edge_4_5_0_0
   + 0.3338008861455377 edge_4_6_0_0 - 0.2864519482758722 edge_4_7_0_0
   + 0.0413303654755953 edge_4_8_0_0 - 0.8408006853768576 edge_4_9_0_0
   - 0.2903178457654434 edge_5_3_0_0 + 0.1371428730820151 edge_5_4_0_0
   + 0.1811798775288171 edge_5_6_0_0 - 0.1806479683234866 edge_5_7_0_0
   - 0.1658566724967296 edge_5_9_0_0 - 0.2482724179588509 edge_6_1_0_0
   - 0.266323897124736 edge_6_2_0_0 - 0.1623778700615103 edge_6_3_0_0
   - 0.0808599020813864 edge_6_4_0_0 + 0.4679773401881689 edge_6_5_0_0
   + 0.3229775868669239 edge_6_7_0_0 + 0.4449070597191588 edge_6_8_0_0
   + 0.049788451077607 edge_6_9_0_0 + 1.930480909930487 edge_6_10_0_0
   - 0.8871787096603748 edge_7_4_0_0 - 0.2594827386076469 edge_7_5_0_0
   - 0.0666875732380375 edge_7_6_0_0 + 0.2608652207178911 edge_8_1_0_0
   + 0.6659317514985299 edge_8_2_0_0 + 0.5799230072873475 edge_8_3_0_0
   + 1.16246852811511 edge_8_4_0_0 + 1.354627513757322 edge_8_5_0_0
   + 2.230584402951475 edge_8_6_0_0 + 3.547005892490485 edge_8_7_0_0
   - 0.0700879140999853 edge_8_8_0_0 + 2.734973016079973 edge_8_9_0_0
   + 2.309757438196374 edge_8_10_0_0 - 0.1516307189002707 edge_9_1_0_0
   - 0.0814557958851603 edge_9_2_0_0 + 0.0897078590533328 edge_9_3_0_0
   - 0.1095081307803301 edge_9_4_0_0 + 0.0457705966941959 edge_9_5_0_0
   - 0.7522845739584936 edge_9_6_0_0 + 0.2641563451050677 edge_9_7_0_0
   - 0.0504778659808696 edge_9_8_0_0 + 0.7817983073065781 edge_9_10_0_0
   - 0.7484513660334366 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_6_1_0_0
   + edge_8_1_0_0 + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_8_0_0 - flow_1_9_0_0 - flow_1_10_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_6_1_0_0 + flow_8_1_0_0
   + flow_9_1_0_0 = 1
 one_parent_2: edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_6_2_0_0
   + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_1_2_0_0 - flow_2_1_0_0 - flow_2_3_0_0
   - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_6_2_0_0
   + flow_8_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_3_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_8_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   - flow_3_10_0_0 + flow_5_3_0_0 + flow_6_3_0_0 + flow_8_3_0_0
   + flow_9_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0 + edge_6_4_0_0
   + edge_7_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 - flow_4_2_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0
   + flow_8_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_1_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0
   - flow_5_7_0_0 - flow_5_9_0_0 + flow_6_5_0_0 + flow_7_5_0_0
   + flow_8_5_0_0 + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_2_7_0_0 + flow_3_7_0_0 + flow_4_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_4_0_0 - flow_7_5_0_0
   - flow_7_6_0_0 + flow_8_7_0_0 + flow_9_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_6_8_0_0 + edge_8_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_6_8_0_0 - flow_8_1_0_0
   - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 + flow_8_9_0_0
   - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0
   - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_1_10_0_0 + edge_3_10_0_0 + edge_6_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_1_10_0_0 + flow_3_10_0_0 + flow_6_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_6_0_0 + flow_0_8_0_0
   + flow_0_9_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_3_0_0: flow_3_3_0_0 - 1000 edge_3_3_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_8_0_0: flow_8_8_0_0 - 1000 edge_8_8_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
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
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_6_0_0 edge_0_8_0_0 edge_0_9_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_5_0_0 edge_1_8_0_0 edge_1_9_0_0
 edge_1_10_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_3_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_4_2_0_0 edge_4_5_0_0
 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_9_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_7_4_0_0 edge_7_5_0_0
 edge_7_6_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_8_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0
 edge_9_5_0_0 edge_9_6_0_0 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0
 edge_10_9_0_0
End
