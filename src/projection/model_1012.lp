Maximize
  - 1.415187433862 edge_0_1_0_0 - 0.3505953065236 edge_0_2_0_0
   - 2.69344252071 edge_0_4_0_0 - 2.0665117832894 edge_0_5_0_0
   - 2.905049128485 edge_0_6_0_0 - 2.9292911886365 edge_0_8_0_0
   - 2.6735029954038 edge_0_9_0_0 - 2.9992999990239 edge_0_10_0_0
   + 0.4860454369036172 edge_1_2_0_0 + 0.2367489931180395 edge_1_4_0_0
   - 0.1167552615918057 edge_1_5_0_0 - 0.2915159402863917 edge_1_6_0_0
   - 0.2130004600511732 edge_1_8_0_0 - 0.4379746039859311 edge_1_9_0_0
   - 0.4026927805988417 edge_1_10_0_0 + 3.9264291325083 edge_2_1_0_0
   + 1.144516299660764 edge_2_3_0_0 + 3.712718886184 edge_2_4_0_0
   + 2.94730055679934 edge_2_5_0_0 + 2.440783627688929 edge_2_6_0_0
   + 1.706148713892497 edge_2_8_0_0 + 1.641579629984 edge_2_9_0_0
   + 3.13125622126605 edge_2_10_0_0 + 0.0234424562668908 edge_3_2_0_0
   + 0.2436628398379665 edge_3_5_0_0 + 0.3907307724482822 edge_3_6_0_0
   + 0.373065865859133 edge_3_8_0_0 + 0.31792469652945 edge_3_9_0_0
   - 0.2271585282828085 edge_3_10_0_0 + 2.42482040531216 edge_4_1_0_0
   + 1.188129643561 edge_4_2_0_0 + 3.3555546698428 edge_4_3_0_0
   + 2.627839762572035 edge_4_5_0_0 + 2.4331050013632 edge_4_6_0_0
   + 3.0076687403766 edge_4_8_0_0 + 2.5517163510404 edge_4_9_0_0
   + 0.5783147363742281 edge_4_10_0_0 - 0.0903173005125103 edge_5_1_0_0
   + 0.1494211243735886 edge_5_2_0_0 - 0.1040198682078012 edge_5_4_0_0
   + 0.44795869617715 edge_5_6_0_0 + 0.1532096575182239 edge_5_8_0_0
   + 0.11538601704726 edge_5_9_0_0 + 0.020637230700447 edge_5_10_0_0
   + 0.91460862354897 edge_6_1_0_0 + 0.626258074616622 edge_6_2_0_0
   + 1.46102712444459 edge_6_3_0_0 + 1.011709696487 edge_6_4_0_0
   + 2.501825169330743 edge_6_5_0_0 + 3.107094538797 edge_6_8_0_0
   + 3.097252777667795 edge_6_9_0_0 + 1.373197871540123 edge_6_10_0_0
   - 0.2876320390640836 edge_8_1_0_0 - 1.078993886672961 edge_8_2_0_0
   - 0.7861289220209861 edge_8_4_0_0 - 0.266739664066596 edge_8_5_0_0
   - 0.0937613777804282 edge_8_6_0_0 + 0.7813080908901 edge_8_9_0_0
   + 0.3227190937052718 edge_8_10_0_0 - 0.0034812651139 edge_9_1_0_0
   - 0.80815856359414 edge_9_2_0_0 + 0.37955030168523 edge_9_3_0_0
   - 0.374719 edge_9_4_0_0 - 0.060052639382984 edge_9_5_0_0
   + 0.128596 edge_9_6_0_0 + 0.5700837219224201 edge_9_8_0_0
   + 0.8396507489185199 edge_9_10_0_0 - 2.021583054008619 edge_10_4_0_0
   - 0.999065781541318 edge_10_5_0_0 - 1.648948433384622 edge_10_6_0_0
   - 0.7777983377029307 edge_10_8_0_0 - 0.506926974331 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_4_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_4_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_8_0_0 - flow_1_9_0_0
   - flow_1_10_0_0 + flow_2_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0
   + flow_6_1_0_0 + flow_8_1_0_0 + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_10_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_8_2_0_0
   + flow_9_2_0_0 = 1
 one_parent_3: edge_2_3_0_0 + edge_4_3_0_0 + edge_6_3_0_0 + edge_9_3_0_0
   = 1
 consume_one_flow_unit_3: flow_2_3_0_0 - flow_3_2_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0
   + flow_4_3_0_0 + flow_6_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0
   - flow_4_6_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_10_0_0
   + flow_5_4_0_0 + flow_6_4_0_0 + flow_8_4_0_0 + flow_9_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_6_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0
   + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_8_0_0 - flow_5_9_0_0
   - flow_5_10_0_0 + flow_6_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0
   + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_9_8_0_0
   + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_4_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_9_0_0 - flow_8_10_0_0 + flow_9_8_0_0
   + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0
   + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0
   - flow_9_4_0_0 - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_4_0_0 - flow_10_5_0_0
   - flow_10_6_0_0 - flow_10_8_0_0 - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_10_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_8_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_1_2_0_0 edge_1_4_0_0
 edge_1_5_0_0 edge_1_6_0_0 edge_1_8_0_0 edge_1_9_0_0 edge_1_10_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0 edge_3_2_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_8_0_0
 edge_4_9_0_0 edge_4_10_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_9_0_0 edge_8_10_0_0 edge_9_1_0_0
 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_8_0_0 edge_9_10_0_0 edge_10_4_0_0 edge_10_5_0_0 edge_10_6_0_0
 edge_10_8_0_0 edge_10_9_0_0
End
