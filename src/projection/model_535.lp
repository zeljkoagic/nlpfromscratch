Maximize
  - 2.3922298123812 edge_0_1_0_0 - 1.5675790620707 edge_0_3_0_0
   - 1.2338613542493 edge_0_5_0_0 + 0.2208544457774962 edge_1_2_0_0
   + 1.012620465735354 edge_2_1_0_0 - 0.5486624632790728 edge_2_2_0_0
   + 0.1286291100563622 edge_2_3_0_0 + 0.4098890047699513 edge_2_4_0_0
   - 2.333796518429451 edge_2_7_0_0 - 2.442855612956663 edge_2_16_0_0
   - 0.3872905545217242 edge_3_1_0_0 + 2.91532049787289 edge_3_2_0_0
   + 0.651468914520076 edge_3_3_0_0 + 1.836257936839813 edge_3_4_0_0
   + 2.072203237597098 edge_3_5_0_0 + 1.104720114020125 edge_3_6_0_0
   - 0.1249511405123407 edge_3_7_0_0 + 0.2395575767866238 edge_3_8_0_0
   - 0.2847016859723405 edge_3_10_0_0 + 1.081488145707408 edge_3_11_0_0
   + 1.02877710863562 edge_3_12_0_0 - 1.262612624886141 edge_3_13_0_0
   - 1.298325898642017 edge_3_16_0_0 - 0.8465809315378975 edge_3_19_0_0
   + 1.894620192439328 edge_3_21_0_0 - 2.40394685331473 edge_5_1_0_0
   + 1.242297325083543 edge_5_2_0_0 + 0.9836768323802135 edge_5_3_0_0
   + 1.167898774282792 edge_5_4_0_0 + 1.388627703700867 edge_5_6_0_0
   + 1.117149803939699 edge_5_7_0_0 + 0.9719405629999345 edge_5_8_0_0
   + 0.6026424752137834 edge_5_10_0_0 + 1.360409600524923 edge_5_11_0_0
   + 1.71301539865704 edge_5_12_0_0 - 0.8932719610006638 edge_5_13_0_0
   - 0.5135342685559179 edge_5_14_0_0 - 0.2961046922556218 edge_5_16_0_0
   + 0.1845222848907482 edge_5_17_0_0 - 0.1263835545705434 edge_5_19_0_0
   + 2.107206701707691 edge_5_21_0_0 - 0.0994571176339356 edge_7_2_0_0
   + 3.087155550505901 edge_7_8_0_0 + 2.944803311455108 edge_7_10_0_0
   + 1.619385960251475 edge_7_11_0_0 + 0.8121310656314219 edge_7_12_0_0
   + 0.7878380424591268 edge_7_13_0_0 + 0.00292427755942894 edge_7_14_0_0
   + 0.5843015751918575 edge_7_16_0_0 + 0.0633370758220243 edge_7_17_0_0
   + 0.3172136584221212 edge_7_19_0_0 + 1.711938373657237 edge_7_21_0_0
   + 0.1000457467725252 edge_10_8_0_0 + 0.9992470409655307 edge_10_11_0_0
   + 0.4585351919332135 edge_10_12_0_0 - 0.1934893777396998 edge_10_13_0_0
   - 0.5767719713364209 edge_10_14_0_0 - 0.074379317376508 edge_10_16_0_0
   - 0.5759750964609646 edge_10_19_0_0 - 0.3943935305576425 edge_11_10_0_0
   - 0.0195143691732738 edge_12_13_0_0 - 0.7930644907213991 edge_12_16_0_0
   - 0.041385838822433 edge_13_12_0_0 + 1.388968266710355 edge_13_14_0_0
   + 1.163248471980879 edge_13_16_0_0 + 0.5262916059127791 edge_13_17_0_0
   + 0.871727343826032 edge_13_19_0_0 - 0.4951213405759243 edge_14_13_0_0
   + 0.0850654401967615 edge_16_10_0_0 + 0.3839535668237218 edge_16_12_0_0
   + 1.594546796491755 edge_16_13_0_0 + 2.326188526669034 edge_16_14_0_0
   + 0.5226099813999452 edge_16_17_0_0 + 2.23179363638448 edge_16_19_0_0
   + 1.955727916956745 edge_16_21_0_0 + 1.064514646272249 edge_17_19_0_0
   + 0.4991019747787931 edge_19_13_0_0 + 0.6832447295036658 edge_19_14_0_0
   + 0.3749565883988895 edge_19_16_0_0 + 1.427904537741586 edge_19_17_0_0
   + 1.939919424508193 edge_19_21_0_0 - 0.4714787221289044 edge_21_17_0_0
   - 0.96144728565663 edge_21_19_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0
   = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_5_1_0_0 = 1
 one_parent_2: edge_1_2_0_0 + edge_2_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0
   + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_1_2_0_0 - flow_2_1_0_0 - flow_2_3_0_0
   - flow_2_4_0_0 - flow_2_7_0_0 - flow_2_16_0_0 + flow_3_2_0_0
   + flow_5_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_3_3_0_0 + edge_5_3_0_0
   = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_1_0_0
   - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_10_0_0 - flow_3_11_0_0
   - flow_3_12_0_0 - flow_3_13_0_0 - flow_3_16_0_0 - flow_3_19_0_0
   - flow_3_21_0_0 + flow_5_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0 = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 + flow_5_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_3_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_3_5_0_0 - flow_5_1_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0
   - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_10_0_0 - flow_5_11_0_0
   - flow_5_12_0_0 - flow_5_13_0_0 - flow_5_14_0_0 - flow_5_16_0_0
   - flow_5_17_0_0 - flow_5_19_0_0 - flow_5_21_0_0 = 1
 one_parent_6: edge_3_6_0_0 + edge_5_6_0_0 = 1
 consume_one_flow_unit_6: flow_3_6_0_0 + flow_5_6_0_0 = 1
 one_parent_7: edge_2_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0 = 1
 consume_one_flow_unit_7: flow_2_7_0_0 + flow_3_7_0_0 + flow_5_7_0_0
   - flow_7_2_0_0 - flow_7_8_0_0 - flow_7_10_0_0 - flow_7_11_0_0
   - flow_7_12_0_0 - flow_7_13_0_0 - flow_7_14_0_0 - flow_7_16_0_0
   - flow_7_17_0_0 - flow_7_19_0_0 - flow_7_21_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_5_8_0_0 + edge_7_8_0_0 + edge_10_8_0_0
   = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_5_8_0_0 + flow_7_8_0_0
   + flow_10_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_3_10_0_0 + edge_5_10_0_0 + edge_7_10_0_0
   + edge_11_10_0_0 + edge_16_10_0_0 = 1
 consume_one_flow_unit_10: flow_3_10_0_0 + flow_5_10_0_0 + flow_7_10_0_0
   - flow_10_8_0_0 - flow_10_11_0_0 - flow_10_12_0_0 - flow_10_13_0_0
   - flow_10_14_0_0 - flow_10_16_0_0 - flow_10_19_0_0 + flow_11_10_0_0
   + flow_16_10_0_0 = 1
 one_parent_11: edge_3_11_0_0 + edge_5_11_0_0 + edge_7_11_0_0
   + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_3_11_0_0 + flow_5_11_0_0 + flow_7_11_0_0
   + flow_10_11_0_0 - flow_11_10_0_0 = 1
 one_parent_12: edge_3_12_0_0 + edge_5_12_0_0 + edge_7_12_0_0
   + edge_10_12_0_0 + edge_13_12_0_0 + edge_16_12_0_0 = 1
 consume_one_flow_unit_12: flow_3_12_0_0 + flow_5_12_0_0 + flow_7_12_0_0
   + flow_10_12_0_0 - flow_12_13_0_0 - flow_12_16_0_0 + flow_13_12_0_0
   + flow_16_12_0_0 = 1
 one_parent_13: edge_3_13_0_0 + edge_5_13_0_0 + edge_7_13_0_0
   + edge_10_13_0_0 + edge_12_13_0_0 + edge_14_13_0_0 + edge_16_13_0_0
   + edge_19_13_0_0 = 1
 consume_one_flow_unit_13: flow_3_13_0_0 + flow_5_13_0_0 + flow_7_13_0_0
   + flow_10_13_0_0 + flow_12_13_0_0 - flow_13_12_0_0 - flow_13_14_0_0
   - flow_13_16_0_0 - flow_13_17_0_0 - flow_13_19_0_0 + flow_14_13_0_0
   + flow_16_13_0_0 + flow_19_13_0_0 = 1
 one_parent_14: edge_5_14_0_0 + edge_7_14_0_0 + edge_10_14_0_0
   + edge_13_14_0_0 + edge_16_14_0_0 + edge_19_14_0_0 = 1
 consume_one_flow_unit_14: flow_5_14_0_0 + flow_7_14_0_0 + flow_10_14_0_0
   + flow_13_14_0_0 - flow_14_13_0_0 + flow_16_14_0_0 + flow_19_14_0_0 = 1
 one_parent_15: = 1
 consume_one_flow_unit_15: = 1
 one_parent_16: edge_2_16_0_0 + edge_3_16_0_0 + edge_5_16_0_0
   + edge_7_16_0_0 + edge_10_16_0_0 + edge_12_16_0_0 + edge_13_16_0_0
   + edge_19_16_0_0 = 1
 consume_one_flow_unit_16: flow_2_16_0_0 + flow_3_16_0_0 + flow_5_16_0_0
   + flow_7_16_0_0 + flow_10_16_0_0 + flow_12_16_0_0 + flow_13_16_0_0
   - flow_16_10_0_0 - flow_16_12_0_0 - flow_16_13_0_0 - flow_16_14_0_0
   - flow_16_17_0_0 - flow_16_19_0_0 - flow_16_21_0_0 + flow_19_16_0_0 = 1
 one_parent_17: edge_5_17_0_0 + edge_7_17_0_0 + edge_13_17_0_0
   + edge_16_17_0_0 + edge_19_17_0_0 + edge_21_17_0_0 = 1
 consume_one_flow_unit_17: flow_5_17_0_0 + flow_7_17_0_0 + flow_13_17_0_0
   + flow_16_17_0_0 - flow_17_19_0_0 + flow_19_17_0_0 + flow_21_17_0_0 = 1
 one_parent_18: = 1
 consume_one_flow_unit_18: = 1
 one_parent_19: edge_3_19_0_0 + edge_5_19_0_0 + edge_7_19_0_0
   + edge_10_19_0_0 + edge_13_19_0_0 + edge_16_19_0_0 + edge_17_19_0_0
   + edge_21_19_0_0 = 1
 consume_one_flow_unit_19: flow_3_19_0_0 + flow_5_19_0_0 + flow_7_19_0_0
   + flow_10_19_0_0 + flow_13_19_0_0 + flow_16_19_0_0 + flow_17_19_0_0
   - flow_19_13_0_0 - flow_19_14_0_0 - flow_19_16_0_0 - flow_19_17_0_0
   - flow_19_21_0_0 + flow_21_19_0_0 = 1
 one_parent_20: = 1
 consume_one_flow_unit_20: = 1
 one_parent_21: edge_3_21_0_0 + edge_5_21_0_0 + edge_7_21_0_0
   + edge_16_21_0_0 + edge_19_21_0_0 = 1
 consume_one_flow_unit_21: flow_3_21_0_0 + flow_5_21_0_0 + flow_7_21_0_0
   + flow_16_21_0_0 + flow_19_21_0_0 - flow_21_17_0_0 - flow_21_19_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_5_0_0 = 21
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_2_0_0: flow_2_2_0_0 - 1000 edge_2_2_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_16_0_0: flow_2_16_0_0 - 1000 edge_2_16_0_0 <= 0
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
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_3_13_0_0: flow_3_13_0_0 - 1000 edge_3_13_0_0 <= 0
 inactive_no_flow_3_16_0_0: flow_3_16_0_0 - 1000 edge_3_16_0_0 <= 0
 inactive_no_flow_3_19_0_0: flow_3_19_0_0 - 1000 edge_3_19_0_0 <= 0
 inactive_no_flow_3_21_0_0: flow_3_21_0_0 - 1000 edge_3_21_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_5_13_0_0: flow_5_13_0_0 - 1000 edge_5_13_0_0 <= 0
 inactive_no_flow_5_14_0_0: flow_5_14_0_0 - 1000 edge_5_14_0_0 <= 0
 inactive_no_flow_5_16_0_0: flow_5_16_0_0 - 1000 edge_5_16_0_0 <= 0
 inactive_no_flow_5_17_0_0: flow_5_17_0_0 - 1000 edge_5_17_0_0 <= 0
 inactive_no_flow_5_19_0_0: flow_5_19_0_0 - 1000 edge_5_19_0_0 <= 0
 inactive_no_flow_5_21_0_0: flow_5_21_0_0 - 1000 edge_5_21_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_7_13_0_0: flow_7_13_0_0 - 1000 edge_7_13_0_0 <= 0
 inactive_no_flow_7_14_0_0: flow_7_14_0_0 - 1000 edge_7_14_0_0 <= 0
 inactive_no_flow_7_16_0_0: flow_7_16_0_0 - 1000 edge_7_16_0_0 <= 0
 inactive_no_flow_7_17_0_0: flow_7_17_0_0 - 1000 edge_7_17_0_0 <= 0
 inactive_no_flow_7_19_0_0: flow_7_19_0_0 - 1000 edge_7_19_0_0 <= 0
 inactive_no_flow_7_21_0_0: flow_7_21_0_0 - 1000 edge_7_21_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_10_13_0_0: flow_10_13_0_0 - 1000 edge_10_13_0_0 <= 0
 inactive_no_flow_10_14_0_0: flow_10_14_0_0 - 1000 edge_10_14_0_0 <= 0
 inactive_no_flow_10_16_0_0: flow_10_16_0_0 - 1000 edge_10_16_0_0 <= 0
 inactive_no_flow_10_19_0_0: flow_10_19_0_0 - 1000 edge_10_19_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_12_13_0_0: flow_12_13_0_0 - 1000 edge_12_13_0_0 <= 0
 inactive_no_flow_12_16_0_0: flow_12_16_0_0 - 1000 edge_12_16_0_0 <= 0
 inactive_no_flow_13_12_0_0: flow_13_12_0_0 - 1000 edge_13_12_0_0 <= 0
 inactive_no_flow_13_14_0_0: flow_13_14_0_0 - 1000 edge_13_14_0_0 <= 0
 inactive_no_flow_13_16_0_0: flow_13_16_0_0 - 1000 edge_13_16_0_0 <= 0
 inactive_no_flow_13_17_0_0: flow_13_17_0_0 - 1000 edge_13_17_0_0 <= 0
 inactive_no_flow_13_19_0_0: flow_13_19_0_0 - 1000 edge_13_19_0_0 <= 0
 inactive_no_flow_14_13_0_0: flow_14_13_0_0 - 1000 edge_14_13_0_0 <= 0
 inactive_no_flow_16_10_0_0: flow_16_10_0_0 - 1000 edge_16_10_0_0 <= 0
 inactive_no_flow_16_12_0_0: flow_16_12_0_0 - 1000 edge_16_12_0_0 <= 0
 inactive_no_flow_16_13_0_0: flow_16_13_0_0 - 1000 edge_16_13_0_0 <= 0
 inactive_no_flow_16_14_0_0: flow_16_14_0_0 - 1000 edge_16_14_0_0 <= 0
 inactive_no_flow_16_17_0_0: flow_16_17_0_0 - 1000 edge_16_17_0_0 <= 0
 inactive_no_flow_16_19_0_0: flow_16_19_0_0 - 1000 edge_16_19_0_0 <= 0
 inactive_no_flow_16_21_0_0: flow_16_21_0_0 - 1000 edge_16_21_0_0 <= 0
 inactive_no_flow_17_19_0_0: flow_17_19_0_0 - 1000 edge_17_19_0_0 <= 0
 inactive_no_flow_19_13_0_0: flow_19_13_0_0 - 1000 edge_19_13_0_0 <= 0
 inactive_no_flow_19_14_0_0: flow_19_14_0_0 - 1000 edge_19_14_0_0 <= 0
 inactive_no_flow_19_16_0_0: flow_19_16_0_0 - 1000 edge_19_16_0_0 <= 0
 inactive_no_flow_19_17_0_0: flow_19_17_0_0 - 1000 edge_19_17_0_0 <= 0
 inactive_no_flow_19_21_0_0: flow_19_21_0_0 - 1000 edge_19_21_0_0 <= 0
 inactive_no_flow_21_17_0_0: flow_21_17_0_0 - 1000 edge_21_17_0_0 <= 0
 inactive_no_flow_21_19_0_0: flow_21_19_0_0 - 1000 edge_21_19_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_1_2_0_0 edge_2_1_0_0
 edge_2_2_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_7_0_0 edge_2_16_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_3_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_10_0_0 edge_3_11_0_0
 edge_3_12_0_0 edge_3_13_0_0 edge_3_16_0_0 edge_3_19_0_0 edge_3_21_0_0
 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_10_0_0 edge_5_11_0_0 edge_5_12_0_0
 edge_5_13_0_0 edge_5_14_0_0 edge_5_16_0_0 edge_5_17_0_0 edge_5_19_0_0
 edge_5_21_0_0 edge_7_2_0_0 edge_7_8_0_0 edge_7_10_0_0 edge_7_11_0_0
 edge_7_12_0_0 edge_7_13_0_0 edge_7_14_0_0 edge_7_16_0_0 edge_7_17_0_0
 edge_7_19_0_0 edge_7_21_0_0 edge_10_8_0_0 edge_10_11_0_0 edge_10_12_0_0
 edge_10_13_0_0 edge_10_14_0_0 edge_10_16_0_0 edge_10_19_0_0 edge_11_10_0_0
 edge_12_13_0_0 edge_12_16_0_0 edge_13_12_0_0 edge_13_14_0_0 edge_13_16_0_0
 edge_13_17_0_0 edge_13_19_0_0 edge_14_13_0_0 edge_16_10_0_0 edge_16_12_0_0
 edge_16_13_0_0 edge_16_14_0_0 edge_16_17_0_0 edge_16_19_0_0 edge_16_21_0_0
 edge_17_19_0_0 edge_19_13_0_0 edge_19_14_0_0 edge_19_16_0_0 edge_19_17_0_0
 edge_19_21_0_0 edge_21_17_0_0 edge_21_19_0_0
End
