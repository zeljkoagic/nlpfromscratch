Maximize
  - 2.8082986609748 edge_0_1_0_0 - 2.8078158326234 edge_0_2_0_0
   - 1.8772521262752 edge_0_3_0_0 - 3.7805309912064 edge_0_4_0_0
   - 3.2147323489188 edge_0_6_0_0 - 3.88365 edge_0_9_0_0
   - 3.3490198189488 edge_0_10_0_0 - 5.0297866803386 edge_0_11_0_0
   + 0.3428176421045145 edge_1_2_0_0 + 0.0174121334138158 edge_1_3_0_0
   + 0.1009169677457912 edge_1_4_0_0 - 0.4340517109358874 edge_1_5_0_0
   - 0.4599642035263588 edge_1_6_0_0 - 0.8060218399371685 edge_1_10_0_0
   - 0.3627740325180605 edge_1_11_0_0 + 0.1085761641441567 edge_2_1_0_0
   + 0.7763288665934273 edge_2_3_0_0 - 0.0250558550082198 edge_2_4_0_0
   + 0.2268606485070128 edge_2_6_0_0 - 0.4262802575712498 edge_2_10_0_0
   + 0.3966081918698607 edge_2_11_0_0 + 0.1035717104220435 edge_3_1_0_0
   + 1.937320400621792 edge_3_2_0_0 + 0.9243885743310497 edge_3_3_0_0
   + 3.599242100893688 edge_3_4_0_0 - 0.2436414787664965 edge_3_5_0_0
   + 3.267469478716737 edge_3_6_0_0 - 0.2602288506803923 edge_3_8_0_0
   + 0.45537589070976 edge_3_9_0_0 + 1.645570488549788 edge_3_10_0_0
   + 1.936118072898576 edge_3_11_0_0 - 0.7818424253823271 edge_4_1_0_0
   - 0.7654579970304799 edge_4_2_0_0 - 0.7173281457559774 edge_4_3_0_0
   - 0.0227462562087795 edge_4_5_0_0 - 0.0838875687621821 edge_4_6_0_0
   - 0.4558865963833341 edge_4_10_0_0 + 0.0440115666785728 edge_4_11_0_0
   - 0.1983395431029494 edge_5_3_0_0 - 0.1972277092789695 edge_5_4_0_0
   + 0.4122957854336953 edge_5_6_0_0 - 0.1294285145259209 edge_5_8_0_0
   - 0.0092577255232 edge_5_9_0_0 + 0.00688144528359274 edge_5_10_0_0
   + 0.0844307294353815 edge_5_11_0_0 + 0.0201504840670805 edge_6_1_0_0
   + 0.626397084696845 edge_6_2_0_0 + 0.6537938947773476 edge_6_3_0_0
   + 0.6480196640773465 edge_6_4_0_0 + 0.7609680655166937 edge_6_5_0_0
   + 0.7382935150857521 edge_6_8_0_0 + 0.893297873946 edge_6_9_0_0
   + 2.192391320702995 edge_6_10_0_0 + 1.543105494120783 edge_6_11_0_0
   - 0.6232959316932581 edge_8_1_0_0 - 1.497767409234235 edge_8_3_0_0
   - 0.7860363400016864 edge_8_6_0_0 + 0.7647136661289999 edge_8_9_0_0
   - 0.1479764205336897 edge_8_11_0_0 - 0.2375161782095 edge_9_1_0_0
   - 1.1686391140455 edge_9_2_0_0 - 0.40691590234016 edge_9_3_0_0
   + 0.0063649073086 edge_9_5_0_0 - 0.562298324346 edge_9_6_0_0
   + 1.8224965573596 edge_9_8_0_0 - 0.55076770258528 edge_9_11_0_0
   + 0.4232017644102863 edge_10_1_0_0 + 0.4995059798142211 edge_10_2_0_0
   + 0.2019678326194486 edge_10_3_0_0 - 0.0722083628631765 edge_10_4_0_0
   + 0.4082316785544244 edge_10_5_0_0 + 1.14055028244763 edge_10_6_0_0
   + 4.240614886277278 edge_10_8_0_0 + 3.6823495996702 edge_10_9_0_0
   + 1.497130717737474 edge_10_11_0_0 - 0.9514641324407581 edge_11_1_0_0
   - 0.9721239727975546 edge_11_10_0_0 + 0 pos_1_0 + 0 pos_2_0 + 0 pos_3_0
   + 0 pos_4_0 + 0 pos_5_0 + 0 pos_6_0 + 0 pos_8_0 + 0 pos_9_0 + 0 pos_10_0
   + 0 pos_11_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_6_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0
   + edge_11_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_10_0_0
   - flow_1_11_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_6_1_0_0 + flow_8_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0
   + flow_11_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_6_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_10_0_0
   - flow_2_11_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_6_2_0_0
   + flow_9_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_3_3_0_0
   + edge_4_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0 + edge_8_3_0_0
   + edge_9_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0
   - flow_3_11_0_0 + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_8_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_10_0_0 - flow_4_11_0_0
   + flow_5_4_0_0 + flow_6_4_0_0 + flow_10_4_0_0 = 1
 one_parent_5: edge_1_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0 + edge_6_5_0_0
   + edge_9_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_1_5_0_0 + flow_3_5_0_0 + flow_4_5_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 - flow_5_10_0_0 - flow_5_11_0_0 + flow_6_5_0_0
   + flow_9_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0 - flow_6_11_0_0
   + flow_8_6_0_0 + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_3_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_9_8_0_0
   + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   - flow_8_1_0_0 - flow_8_3_0_0 - flow_8_6_0_0 - flow_8_9_0_0
   - flow_8_11_0_0 + flow_9_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0
   + edge_8_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_3_9_0_0 + flow_5_9_0_0
   + flow_6_9_0_0 + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0
   - flow_9_3_0_0 - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_8_0_0
   - flow_9_11_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0
   - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_8_0_0 - flow_10_9_0_0
   - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_4_11_0_0 + edge_5_11_0_0 + edge_6_11_0_0
   + edge_8_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_4_11_0_0 + flow_5_11_0_0 + flow_6_11_0_0
   + flow_8_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_1_0_0
   - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_6_0_0 + flow_0_9_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
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
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_3_0_0: flow_3_3_0_0 - 1000 edge_3_3_0_0 <= 0
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
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
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
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
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
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_1_6_0_0 edge_1_10_0_0 edge_1_11_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_6_0_0 edge_2_10_0_0
 edge_2_11_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_3_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0
 edge_3_11_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0
 edge_4_6_0_0 edge_4_10_0_0 edge_4_11_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0
 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_8_1_0_0
 edge_8_3_0_0 edge_8_6_0_0 edge_8_9_0_0 edge_8_11_0_0 edge_9_1_0_0
 edge_9_2_0_0 edge_9_3_0_0 edge_9_5_0_0 edge_9_6_0_0 edge_9_8_0_0
 edge_9_11_0_0 edge_10_1_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0
 edge_10_5_0_0 edge_10_6_0_0 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0
 edge_11_1_0_0 edge_11_10_0_0 pos_1_0 pos_2_0 pos_3_0 pos_4_0 pos_5_0
 pos_6_0 pos_8_0 pos_9_0 pos_10_0 pos_11_0
End
