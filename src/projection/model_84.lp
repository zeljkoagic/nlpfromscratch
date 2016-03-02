Maximize
  - 2.5184182089955 edge_0_1_0_0 - 1.3458188989944 edge_0_3_0_0
   - 1.1149556805088 edge_0_5_0_0 - 0.702317335314 edge_0_7_0_0
   - 0.8758297616094001 edge_0_8_0_0 - 2.534484976246 edge_0_12_0_0
   - 1.8763892826976 edge_0_14_0_0 - 4.8598133420466 edge_0_15_0_0
   - 0.1668747523493912 edge_1_3_0_0 - 0.0471371757942465 edge_1_5_0_0
   - 0.0776413177578536 edge_1_7_0_0 - 0.5961192288165589 edge_1_12_0_0
   - 0.4013720966042743 edge_1_15_0_0 + 0.3640054951472623 edge_3_1_0_0
   + 0.3528014055573008 edge_3_7_0_0 - 0.0848603717604325 edge_3_8_0_0
   - 0.1864352321443754 edge_3_9_0_0 - 0.4898285193679743 edge_3_10_0_0
   + 0.2143123381398336 edge_3_12_0_0 + 0.5957366739052385 edge_3_15_0_0
   - 0.0883292235838872 edge_5_1_0_0 + 0.2368247559578595 edge_5_6_0_0
   + 0.0469171609879532 edge_5_7_0_0 - 0.1011810768485519 edge_5_8_0_0
   + 0.0246947044208361 edge_5_12_0_0 - 0.0274239165855337 edge_5_14_0_0
   - 0.2232224437855724 edge_5_15_0_0 - 0.9113561684360624 edge_6_1_0_0
   - 0.5065027644785931 edge_6_3_0_0 - 0.0169725319630375 edge_6_5_0_0
   - 0.5492959174202996 edge_6_7_0_0 + 0.2211084905611711 edge_6_12_0_0
   - 0.611852133037143 edge_6_15_0_0 + 0.1650442621881267 edge_7_1_0_0
   + 0.058473651163973 edge_7_3_0_0 + 0.1292093174549356 edge_7_5_0_0
   + 2.138971437654319 edge_7_6_0_0 + 0.0735790325153969 edge_7_7_0_0
   + 0.1731094519202163 edge_7_8_0_0 + 0.9848591147996256 edge_7_12_0_0
   + 0.0913226309201849 edge_7_14_0_0 + 0.4111225366968731 edge_7_15_0_0
   + 0.0392488384063801 edge_8_5_0_0 + 0.1203906195903054 edge_8_6_0_0
   + 0.4621984570810977 edge_8_7_0_0 + 0.1979804591719704 edge_8_12_0_0
   + 1.073947618730616 edge_8_15_0_0 + 0.0531146101462468 edge_9_3_0_0
   + 0.298520674861661 edge_9_8_0_0 - 0.0553217027312829 edge_9_10_0_0
   + 0.4281923068102716 edge_9_12_0_0 + 0.1941092701358962 edge_9_15_0_0
   - 0.0095589992218718 edge_10_9_0_0 + 0.27151158735405 edge_12_1_0_0
   + 0.0603606995816106 edge_12_3_0_0 + 0.1304228579331354 edge_12_5_0_0
   + 1.136079278471093 edge_12_6_0_0 + 0.378267084075488 edge_12_7_0_0
   + 0.4612245741294139 edge_12_8_0_0 + 0.3599778209155057 edge_12_9_0_0
   + 1.6232238223303 edge_12_10_0_0 - 0.092135448100636 edge_12_14_0_0
   + 0.4391784025877687 edge_12_15_0_0 - 0.3922047635655915 edge_14_1_0_0
   - 0.00659291944221464 edge_14_3_0_0 + 0.3636551964453192 edge_14_5_0_0
   + 0.00512510375419742 edge_14_7_0_0 + 0.0234549733985421 edge_14_8_0_0
   - 0.1057098475570398 edge_14_9_0_0 + 0.4603946305827469 edge_14_12_0_0
   + 0.2869720070649532 edge_14_15_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0 + edge_6_1_0_0
   + edge_7_1_0_0 + edge_12_1_0_0 + edge_14_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_5_0_0
   - flow_1_7_0_0 - flow_1_12_0_0 - flow_1_15_0_0 + flow_3_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_12_1_0_0
   + flow_14_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0
   + edge_9_3_0_0 + edge_12_3_0_0 + edge_14_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0
   - flow_3_12_0_0 - flow_3_15_0_0 + flow_6_3_0_0 + flow_7_3_0_0
   + flow_9_3_0_0 + flow_12_3_0_0 + flow_14_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_6_5_0_0 + edge_7_5_0_0
   + edge_8_5_0_0 + edge_12_5_0_0 + edge_14_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 - flow_5_1_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_12_0_0
   - flow_5_14_0_0 - flow_5_15_0_0 + flow_6_5_0_0 + flow_7_5_0_0
   + flow_8_5_0_0 + flow_12_5_0_0 + flow_14_5_0_0 = 1
 one_parent_6: edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_12_6_0_0
   = 1
 consume_one_flow_unit_6: flow_5_6_0_0 - flow_6_1_0_0 - flow_6_3_0_0
   - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_12_0_0 - flow_6_15_0_0
   + flow_7_6_0_0 + flow_8_6_0_0 + flow_12_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_7_7_0_0 + edge_8_7_0_0 + edge_12_7_0_0
   + edge_14_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_3_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_3_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_12_0_0
   - flow_7_14_0_0 - flow_7_15_0_0 + flow_8_7_0_0 + flow_12_7_0_0
   + flow_14_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0 + edge_7_8_0_0
   + edge_9_8_0_0 + edge_12_8_0_0 + edge_14_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_3_8_0_0 + flow_5_8_0_0
   + flow_7_8_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0
   - flow_8_12_0_0 - flow_8_15_0_0 + flow_9_8_0_0 + flow_12_8_0_0
   + flow_14_8_0_0 = 1
 one_parent_9: edge_3_9_0_0 + edge_10_9_0_0 + edge_12_9_0_0 + edge_14_9_0_0
   = 1
 consume_one_flow_unit_9: flow_3_9_0_0 - flow_9_3_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 - flow_9_12_0_0 - flow_9_15_0_0 + flow_10_9_0_0
   + flow_12_9_0_0 + flow_14_9_0_0 = 1
 one_parent_10: edge_3_10_0_0 + edge_9_10_0_0 + edge_12_10_0_0 = 1
 consume_one_flow_unit_10: flow_3_10_0_0 + flow_9_10_0_0 - flow_10_9_0_0
   + flow_12_10_0_0 = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_3_12_0_0
   + edge_5_12_0_0 + edge_6_12_0_0 + edge_7_12_0_0 + edge_8_12_0_0
   + edge_9_12_0_0 + edge_14_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_3_12_0_0
   + flow_5_12_0_0 + flow_6_12_0_0 + flow_7_12_0_0 + flow_8_12_0_0
   + flow_9_12_0_0 - flow_12_1_0_0 - flow_12_3_0_0 - flow_12_5_0_0
   - flow_12_6_0_0 - flow_12_7_0_0 - flow_12_8_0_0 - flow_12_9_0_0
   - flow_12_10_0_0 - flow_12_14_0_0 - flow_12_15_0_0 + flow_14_12_0_0 = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: edge_0_14_0_0 + edge_5_14_0_0 + edge_7_14_0_0
   + edge_12_14_0_0 = 1
 consume_one_flow_unit_14: flow_0_14_0_0 + flow_5_14_0_0 + flow_7_14_0_0
   + flow_12_14_0_0 - flow_14_1_0_0 - flow_14_3_0_0 - flow_14_5_0_0
   - flow_14_7_0_0 - flow_14_8_0_0 - flow_14_9_0_0 - flow_14_12_0_0
   - flow_14_15_0_0 = 1
 one_parent_15: edge_0_15_0_0 + edge_1_15_0_0 + edge_3_15_0_0
   + edge_5_15_0_0 + edge_6_15_0_0 + edge_7_15_0_0 + edge_8_15_0_0
   + edge_9_15_0_0 + edge_12_15_0_0 + edge_14_15_0_0 = 1
 consume_one_flow_unit_15: flow_0_15_0_0 + flow_1_15_0_0 + flow_3_15_0_0
   + flow_5_15_0_0 + flow_6_15_0_0 + flow_7_15_0_0 + flow_8_15_0_0
   + flow_9_15_0_0 + flow_12_15_0_0 + flow_14_15_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_7_0_0
   + flow_0_8_0_0 + flow_0_12_0_0 + flow_0_14_0_0 + flow_0_15_0_0 = 15
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_0_14_0_0: flow_0_14_0_0 - 1000 edge_0_14_0_0 <= 0
 inactive_no_flow_0_15_0_0: flow_0_15_0_0 - 1000 edge_0_15_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_1_15_0_0: flow_1_15_0_0 - 1000 edge_1_15_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_3_15_0_0: flow_3_15_0_0 - 1000 edge_3_15_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_5_14_0_0: flow_5_14_0_0 - 1000 edge_5_14_0_0 <= 0
 inactive_no_flow_5_15_0_0: flow_5_15_0_0 - 1000 edge_5_15_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_6_15_0_0: flow_6_15_0_0 - 1000 edge_6_15_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_7_0_0: flow_7_7_0_0 - 1000 edge_7_7_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_7_14_0_0: flow_7_14_0_0 - 1000 edge_7_14_0_0 <= 0
 inactive_no_flow_7_15_0_0: flow_7_15_0_0 - 1000 edge_7_15_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_8_15_0_0: flow_8_15_0_0 - 1000 edge_8_15_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_15_0_0: flow_9_15_0_0 - 1000 edge_9_15_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_5_0_0: flow_12_5_0_0 - 1000 edge_12_5_0_0 <= 0
 inactive_no_flow_12_6_0_0: flow_12_6_0_0 - 1000 edge_12_6_0_0 <= 0
 inactive_no_flow_12_7_0_0: flow_12_7_0_0 - 1000 edge_12_7_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_14_0_0: flow_12_14_0_0 - 1000 edge_12_14_0_0 <= 0
 inactive_no_flow_12_15_0_0: flow_12_15_0_0 - 1000 edge_12_15_0_0 <= 0
 inactive_no_flow_14_1_0_0: flow_14_1_0_0 - 1000 edge_14_1_0_0 <= 0
 inactive_no_flow_14_3_0_0: flow_14_3_0_0 - 1000 edge_14_3_0_0 <= 0
 inactive_no_flow_14_5_0_0: flow_14_5_0_0 - 1000 edge_14_5_0_0 <= 0
 inactive_no_flow_14_7_0_0: flow_14_7_0_0 - 1000 edge_14_7_0_0 <= 0
 inactive_no_flow_14_8_0_0: flow_14_8_0_0 - 1000 edge_14_8_0_0 <= 0
 inactive_no_flow_14_9_0_0: flow_14_9_0_0 - 1000 edge_14_9_0_0 <= 0
 inactive_no_flow_14_12_0_0: flow_14_12_0_0 - 1000 edge_14_12_0_0 <= 0
 inactive_no_flow_14_15_0_0: flow_14_15_0_0 - 1000 edge_14_15_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_7_0_0 edge_0_8_0_0
 edge_0_12_0_0 edge_0_14_0_0 edge_0_15_0_0 edge_1_3_0_0 edge_1_5_0_0
 edge_1_7_0_0 edge_1_12_0_0 edge_1_15_0_0 edge_3_1_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_3_12_0_0 edge_3_15_0_0
 edge_5_1_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_12_0_0
 edge_5_14_0_0 edge_5_15_0_0 edge_6_1_0_0 edge_6_3_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_6_12_0_0 edge_6_15_0_0 edge_7_1_0_0 edge_7_3_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_7_0_0 edge_7_8_0_0 edge_7_12_0_0
 edge_7_14_0_0 edge_7_15_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0
 edge_8_12_0_0 edge_8_15_0_0 edge_9_3_0_0 edge_9_8_0_0 edge_9_10_0_0
 edge_9_12_0_0 edge_9_15_0_0 edge_10_9_0_0 edge_12_1_0_0 edge_12_3_0_0
 edge_12_5_0_0 edge_12_6_0_0 edge_12_7_0_0 edge_12_8_0_0 edge_12_9_0_0
 edge_12_10_0_0 edge_12_14_0_0 edge_12_15_0_0 edge_14_1_0_0 edge_14_3_0_0
 edge_14_5_0_0 edge_14_7_0_0 edge_14_8_0_0 edge_14_9_0_0 edge_14_12_0_0
 edge_14_15_0_0
End
