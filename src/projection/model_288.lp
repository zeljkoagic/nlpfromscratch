Maximize
  - 3.021574109368 edge_0_1_0_0 - 3.7484575260164 edge_0_2_0_0
   - 2.4738402208215 edge_0_3_0_0 - 3.4597716591692 edge_0_4_0_0
   - 2.5648706520477 edge_0_5_0_0 - 0.3504523757454794 edge_1_2_0_0
   - 0.00255219962461352 edge_1_3_0_0 - 0.5454679410869693 edge_1_4_0_0
   - 0.6339285112363982 edge_1_5_0_0 - 1.319756990502961 edge_1_6_0_0
   - 1.365749532321438 edge_1_8_0_0 + 0.325290806613726 edge_2_1_0_0
   + 0.7884141479546189 edge_2_3_0_0 + 0.0911928171834009 edge_2_4_0_0
   - 0.00965585989748826 edge_2_5_0_0 - 0.6135008475184235 edge_2_6_0_0
   - 0.5738976114420464 edge_2_7_0_0 - 0.608767269468707 edge_2_8_0_0
   - 0.2555018160385701 edge_2_10_0_0 + 0.4267194879361521 edge_2_11_0_0
   + 0.5056856714228611 edge_3_1_0_0 + 0.856920017883815 edge_3_2_0_0
   + 0.1221109471756178 edge_3_3_0_0 + 0.2445401378292521 edge_3_4_0_0
   + 0.1985935411825802 edge_3_5_0_0 - 0.3332896888493495 edge_3_6_0_0
   - 0.3903960140134983 edge_3_7_0_0 - 0.1895725850515757 edge_3_8_0_0
   - 0.3765599031052845 edge_3_10_0_0 + 1.578171693287825 edge_3_11_0_0
   - 0.6999117481958473 edge_4_1_0_0 - 0.693197395489897 edge_4_2_0_0
   - 0.4363962467993557 edge_4_3_0_0 + 0.0118730230753623 edge_4_5_0_0
   - 0.0896520632271146 edge_4_6_0_0 - 0.0531619430070003 edge_5_1_0_0
   - 0.4214492377271295 edge_5_2_0_0 - 0.2008064231372242 edge_5_3_0_0
   + 0.0840337304651639 edge_5_4_0_0 + 2.344010369086405 edge_5_6_0_0
   + 1.326949965383891 edge_5_7_0_0 + 0.7550329122711781 edge_5_8_0_0
   + 1.305654363970533 edge_5_10_0_0 + 1.89040399811909 edge_5_11_0_0
   - 0.7052034339232739 edge_6_1_0_0 - 0.9093594571215432 edge_6_3_0_0
   - 0.2111029451137103 edge_6_4_0_0 + 0.0566575142498271 edge_6_7_0_0
   - 0.2180092954410278 edge_6_8_0_0 - 0.038749564562737 edge_6_10_0_0
   + 0.1311827747381134 edge_6_11_0_0 + 0.0917548776537978 edge_7_1_0_0
   - 0.2738066485996527 edge_7_2_0_0 - 0.1152304411517724 edge_7_3_0_0
   - 0.2258377656491569 edge_7_4_0_0 - 0.1870162468321082 edge_7_5_0_0
   + 0.2602294942516855 edge_7_6_0_0 + 0.8952795782142557 edge_7_8_0_0
   + 0.9790741584923471 edge_7_10_0_0 + 0.4604827355369485 edge_7_11_0_0
   + 0.9199782443589271 edge_8_7_0_0 + 0.6475431459576889 edge_8_10_0_0
   - 0.3475379765586913 edge_8_11_0_0 - 0.3506531004566127 edge_10_1_0_0
   - 0.5879285766911434 edge_10_2_0_0 - 0.6321831819300467 edge_10_3_0_0
   - 0.1415511474616407 edge_10_5_0_0 + 0.3275831775954782 edge_10_6_0_0
   + 0.8450874730912286 edge_10_7_0_0 + 2.776329400959226 edge_10_8_0_0
   + 0.92804444317303 edge_10_11_0_0 - 1.299557308217875 edge_11_7_0_0
   - 1.131249049850983 edge_11_8_0_0 - 1.513349091990049 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_8_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0
   + flow_6_1_0_0 + flow_7_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_7_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_10_0_0 - flow_2_11_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0 + flow_7_2_0_0
   + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_3_3_0_0
   + edge_4_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0
   + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_10_0_0
   - flow_3_11_0_0 + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_7_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 + flow_5_4_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_7_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_8_0_0 - flow_5_10_0_0 - flow_5_11_0_0 + flow_7_5_0_0
   + flow_10_5_0_0 = 1
 one_parent_6: edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_1_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 + flow_7_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_2_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0
   + edge_8_7_0_0 + edge_10_7_0_0 + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_2_7_0_0 + flow_3_7_0_0 + flow_5_7_0_0
   + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0
   - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0
   - flow_7_10_0_0 - flow_7_11_0_0 + flow_8_7_0_0 + flow_10_7_0_0
   + flow_11_7_0_0 = 1
 one_parent_8: edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_1_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_7_0_0
   - flow_8_10_0_0 - flow_8_11_0_0 + flow_10_8_0_0 + flow_11_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_2_10_0_0 + edge_3_10_0_0 + edge_5_10_0_0
   + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_3_10_0_0 + flow_5_10_0_0
   + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0 - flow_10_1_0_0
   - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_5_0_0 - flow_10_6_0_0
   - flow_10_7_0_0 - flow_10_8_0_0 - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_2_11_0_0 + edge_3_11_0_0 + edge_5_11_0_0
   + edge_6_11_0_0 + edge_7_11_0_0 + edge_8_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_2_11_0_0 + flow_3_11_0_0 + flow_5_11_0_0
   + flow_6_11_0_0 + flow_7_11_0_0 + flow_8_11_0_0 + flow_10_11_0_0
   - flow_11_7_0_0 - flow_11_8_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_3_0_0: flow_3_3_0_0 - 1000 edge_3_3_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_5_0_0 edge_1_6_0_0
 edge_1_8_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_10_0_0 edge_2_11_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_3_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_10_0_0 edge_3_11_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0
 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_10_0_0 edge_5_11_0_0 edge_6_1_0_0
 edge_6_3_0_0 edge_6_4_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_6_10_0_0
 edge_6_11_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_10_0_0 edge_7_11_0_0
 edge_8_7_0_0 edge_8_10_0_0 edge_8_11_0_0 edge_10_1_0_0 edge_10_2_0_0
 edge_10_3_0_0 edge_10_5_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_8_0_0
 edge_10_11_0_0 edge_11_7_0_0 edge_11_8_0_0 edge_11_10_0_0
End
