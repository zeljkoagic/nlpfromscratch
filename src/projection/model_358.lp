Maximize
  - 2.772001143366 edge_0_1_0_0 - 1.1778099601187 edge_0_3_0_0
   - 1.9361647642954 edge_0_5_0_0 - 1.8827703358281 edge_0_26_0_0
   - 2.8818595768026 edge_0_27_0_0 - 2.5619236992402 edge_0_31_0_0
   - 0.4119029322525 edge_0_33_0_0 - 1.464870415808 edge_0_34_0_0
   + 0.0102165286730614 edge_1_3_0_0 - 0.2583574141296489 edge_1_5_0_0
   - 0.6050057013160791 edge_1_26_0_0 - 0.1249078292386822 edge_1_27_0_0
   - 0.8087428075614608 edge_1_31_0_0 - 0.2360383614579389 edge_1_33_0_0
   - 0.6766669510523941 edge_1_34_0_0 - 0.3728182000344187 edge_1_38_0_0
   - 0.0486124755681839 edge_3_1_0_0 + 0.1092936684278259 edge_3_5_0_0
   + 0.0689930753832089 edge_3_26_0_0 + 0.1870603995366005 edge_3_27_0_0
   - 0.1465677585184172 edge_3_31_0_0 - 0.0271959640568394 edge_3_33_0_0
   - 0.1822324876376449 edge_3_34_0_0 + 0.1980009899302712 edge_3_38_0_0
   - 0.0852373255192123 edge_5_1_0_0 + 0.0888846378254203 edge_5_3_0_0
   + 0.2809324964524306 edge_5_26_0_0 - 0.2656932523175388 edge_5_27_0_0
   - 0.5632202252648358 edge_5_31_0_0 - 0.0348585366988599 edge_5_33_0_0
   - 0.3150194217333885 edge_5_34_0_0 - 0.5415222877136138 edge_5_38_0_0
   + 0.1409444917320367 edge_26_1_0_0 + 0.4766334683820797 edge_26_3_0_0
   + 1.61073996317909 edge_26_5_0_0 + 1.719190791081007 edge_26_27_0_0
   + 0.3206588671673776 edge_26_31_0_0 + 0.5115920446268049 edge_26_33_0_0
   + 0.6718317695007321 edge_26_34_0_0 + 1.324496539468094 edge_26_38_0_0
   - 0.6699315096862929 edge_27_1_0_0 - 0.3467703595998697 edge_27_3_0_0
   - 0.3652017103083933 edge_27_5_0_0 - 0.352185215664149 edge_27_26_0_0
   + 0.1737804910530775 edge_27_31_0_0 + 0.0742108246561043 edge_27_33_0_0
   - 0.0372308634076986 edge_27_34_0_0 + 0.261880857634893 edge_27_38_0_0
   - 0.9030063821949973 edge_31_1_0_0 - 0.3751690415406848 edge_31_3_0_0
   - 0.6989548441402867 edge_31_26_0_0 + 0.199692046261098 edge_31_27_0_0
   + 0.2086125199929593 edge_31_33_0_0 + 0.0456105858121413 edge_31_34_0_0
   + 0.126290954718317 edge_31_38_0_0 + 0.0457944738562356 edge_33_1_0_0
   - 0.0117139536652943 edge_33_3_0_0 - 0.0274474039072883 edge_33_5_0_0
   - 0.0193064947520338 edge_33_26_0_0 + 0.2069429486120432 edge_33_27_0_0
   + 0.7980715960185878 edge_33_31_0_0 + 0.4829279566614707 edge_33_34_0_0
   + 0.5214050300168255 edge_33_38_0_0 - 0.4643810277488605 edge_34_1_0_0
   - 0.118070245205741 edge_34_3_0_0 - 0.123512721941033 edge_34_5_0_0
   - 0.5175188094122358 edge_34_26_0_0 + 0.0790990558972308 edge_34_27_0_0
   + 0.353053502674213 edge_34_31_0_0 + 0.1080358894014503 edge_34_33_0_0
   + 0.4612593061983151 edge_34_38_0_0 - 0.9761174074554209 edge_38_31_0_0
   - 0.683730534004926 edge_38_34_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0 + edge_26_1_0_0
   + edge_27_1_0_0 + edge_31_1_0_0 + edge_33_1_0_0 + edge_34_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_5_0_0
   - flow_1_26_0_0 - flow_1_27_0_0 - flow_1_31_0_0 - flow_1_33_0_0
   - flow_1_34_0_0 - flow_1_38_0_0 + flow_3_1_0_0 + flow_5_1_0_0
   + flow_26_1_0_0 + flow_27_1_0_0 + flow_31_1_0_0 + flow_33_1_0_0
   + flow_34_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_5_3_0_0 + edge_26_3_0_0
   + edge_27_3_0_0 + edge_31_3_0_0 + edge_33_3_0_0 + edge_34_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_5_0_0 - flow_3_26_0_0 - flow_3_27_0_0 - flow_3_31_0_0
   - flow_3_33_0_0 - flow_3_34_0_0 - flow_3_38_0_0 + flow_5_3_0_0
   + flow_26_3_0_0 + flow_27_3_0_0 + flow_31_3_0_0 + flow_33_3_0_0
   + flow_34_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_26_5_0_0
   + edge_27_5_0_0 + edge_33_5_0_0 + edge_34_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   - flow_5_1_0_0 - flow_5_3_0_0 - flow_5_26_0_0 - flow_5_27_0_0
   - flow_5_31_0_0 - flow_5_33_0_0 - flow_5_34_0_0 - flow_5_38_0_0
   + flow_26_5_0_0 + flow_27_5_0_0 + flow_33_5_0_0 + flow_34_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: = 1
 consume_one_flow_unit_14: = 1
 one_parent_15: = 1
 consume_one_flow_unit_15: = 1
 one_parent_16: = 1
 consume_one_flow_unit_16: = 1
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
 one_parent_23: = 1
 consume_one_flow_unit_23: = 1
 one_parent_24: = 1
 consume_one_flow_unit_24: = 1
 one_parent_25: = 1
 consume_one_flow_unit_25: = 1
 one_parent_26: edge_0_26_0_0 + edge_1_26_0_0 + edge_3_26_0_0
   + edge_5_26_0_0 + edge_27_26_0_0 + edge_31_26_0_0 + edge_33_26_0_0
   + edge_34_26_0_0 = 1
 consume_one_flow_unit_26: flow_0_26_0_0 + flow_1_26_0_0 + flow_3_26_0_0
   + flow_5_26_0_0 - flow_26_1_0_0 - flow_26_3_0_0 - flow_26_5_0_0
   - flow_26_27_0_0 - flow_26_31_0_0 - flow_26_33_0_0 - flow_26_34_0_0
   - flow_26_38_0_0 + flow_27_26_0_0 + flow_31_26_0_0 + flow_33_26_0_0
   + flow_34_26_0_0 = 1
 one_parent_27: edge_0_27_0_0 + edge_1_27_0_0 + edge_3_27_0_0
   + edge_5_27_0_0 + edge_26_27_0_0 + edge_31_27_0_0 + edge_33_27_0_0
   + edge_34_27_0_0 = 1
 consume_one_flow_unit_27: flow_0_27_0_0 + flow_1_27_0_0 + flow_3_27_0_0
   + flow_5_27_0_0 + flow_26_27_0_0 - flow_27_1_0_0 - flow_27_3_0_0
   - flow_27_5_0_0 - flow_27_26_0_0 - flow_27_31_0_0 - flow_27_33_0_0
   - flow_27_34_0_0 - flow_27_38_0_0 + flow_31_27_0_0 + flow_33_27_0_0
   + flow_34_27_0_0 = 1
 one_parent_28: = 1
 consume_one_flow_unit_28: = 1
 one_parent_29: = 1
 consume_one_flow_unit_29: = 1
 one_parent_30: = 1
 consume_one_flow_unit_30: = 1
 one_parent_31: edge_0_31_0_0 + edge_1_31_0_0 + edge_3_31_0_0
   + edge_5_31_0_0 + edge_26_31_0_0 + edge_27_31_0_0 + edge_33_31_0_0
   + edge_34_31_0_0 + edge_38_31_0_0 = 1
 consume_one_flow_unit_31: flow_0_31_0_0 + flow_1_31_0_0 + flow_3_31_0_0
   + flow_5_31_0_0 + flow_26_31_0_0 + flow_27_31_0_0 - flow_31_1_0_0
   - flow_31_3_0_0 - flow_31_26_0_0 - flow_31_27_0_0 - flow_31_33_0_0
   - flow_31_34_0_0 - flow_31_38_0_0 + flow_33_31_0_0 + flow_34_31_0_0
   + flow_38_31_0_0 = 1
 one_parent_32: = 1
 consume_one_flow_unit_32: = 1
 one_parent_33: edge_0_33_0_0 + edge_1_33_0_0 + edge_3_33_0_0
   + edge_5_33_0_0 + edge_26_33_0_0 + edge_27_33_0_0 + edge_31_33_0_0
   + edge_34_33_0_0 = 1
 consume_one_flow_unit_33: flow_0_33_0_0 + flow_1_33_0_0 + flow_3_33_0_0
   + flow_5_33_0_0 + flow_26_33_0_0 + flow_27_33_0_0 + flow_31_33_0_0
   - flow_33_1_0_0 - flow_33_3_0_0 - flow_33_5_0_0 - flow_33_26_0_0
   - flow_33_27_0_0 - flow_33_31_0_0 - flow_33_34_0_0 - flow_33_38_0_0
   + flow_34_33_0_0 = 1
 one_parent_34: edge_0_34_0_0 + edge_1_34_0_0 + edge_3_34_0_0
   + edge_5_34_0_0 + edge_26_34_0_0 + edge_27_34_0_0 + edge_31_34_0_0
   + edge_33_34_0_0 + edge_38_34_0_0 = 1
 consume_one_flow_unit_34: flow_0_34_0_0 + flow_1_34_0_0 + flow_3_34_0_0
   + flow_5_34_0_0 + flow_26_34_0_0 + flow_27_34_0_0 + flow_31_34_0_0
   + flow_33_34_0_0 - flow_34_1_0_0 - flow_34_3_0_0 - flow_34_5_0_0
   - flow_34_26_0_0 - flow_34_27_0_0 - flow_34_31_0_0 - flow_34_33_0_0
   - flow_34_38_0_0 + flow_38_34_0_0 = 1
 one_parent_35: = 1
 consume_one_flow_unit_35: = 1
 one_parent_36: = 1
 consume_one_flow_unit_36: = 1
 one_parent_37: = 1
 consume_one_flow_unit_37: = 1
 one_parent_38: edge_1_38_0_0 + edge_3_38_0_0 + edge_5_38_0_0
   + edge_26_38_0_0 + edge_27_38_0_0 + edge_31_38_0_0 + edge_33_38_0_0
   + edge_34_38_0_0 = 1
 consume_one_flow_unit_38: flow_1_38_0_0 + flow_3_38_0_0 + flow_5_38_0_0
   + flow_26_38_0_0 + flow_27_38_0_0 + flow_31_38_0_0 + flow_33_38_0_0
   + flow_34_38_0_0 - flow_38_31_0_0 - flow_38_34_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_26_0_0
   + flow_0_27_0_0 + flow_0_31_0_0 + flow_0_33_0_0 + flow_0_34_0_0 = 38
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_26_0_0: flow_0_26_0_0 - 1000 edge_0_26_0_0 <= 0
 inactive_no_flow_0_27_0_0: flow_0_27_0_0 - 1000 edge_0_27_0_0 <= 0
 inactive_no_flow_0_31_0_0: flow_0_31_0_0 - 1000 edge_0_31_0_0 <= 0
 inactive_no_flow_0_33_0_0: flow_0_33_0_0 - 1000 edge_0_33_0_0 <= 0
 inactive_no_flow_0_34_0_0: flow_0_34_0_0 - 1000 edge_0_34_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_26_0_0: flow_1_26_0_0 - 1000 edge_1_26_0_0 <= 0
 inactive_no_flow_1_27_0_0: flow_1_27_0_0 - 1000 edge_1_27_0_0 <= 0
 inactive_no_flow_1_31_0_0: flow_1_31_0_0 - 1000 edge_1_31_0_0 <= 0
 inactive_no_flow_1_33_0_0: flow_1_33_0_0 - 1000 edge_1_33_0_0 <= 0
 inactive_no_flow_1_34_0_0: flow_1_34_0_0 - 1000 edge_1_34_0_0 <= 0
 inactive_no_flow_1_38_0_0: flow_1_38_0_0 - 1000 edge_1_38_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_26_0_0: flow_3_26_0_0 - 1000 edge_3_26_0_0 <= 0
 inactive_no_flow_3_27_0_0: flow_3_27_0_0 - 1000 edge_3_27_0_0 <= 0
 inactive_no_flow_3_31_0_0: flow_3_31_0_0 - 1000 edge_3_31_0_0 <= 0
 inactive_no_flow_3_33_0_0: flow_3_33_0_0 - 1000 edge_3_33_0_0 <= 0
 inactive_no_flow_3_34_0_0: flow_3_34_0_0 - 1000 edge_3_34_0_0 <= 0
 inactive_no_flow_3_38_0_0: flow_3_38_0_0 - 1000 edge_3_38_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_26_0_0: flow_5_26_0_0 - 1000 edge_5_26_0_0 <= 0
 inactive_no_flow_5_27_0_0: flow_5_27_0_0 - 1000 edge_5_27_0_0 <= 0
 inactive_no_flow_5_31_0_0: flow_5_31_0_0 - 1000 edge_5_31_0_0 <= 0
 inactive_no_flow_5_33_0_0: flow_5_33_0_0 - 1000 edge_5_33_0_0 <= 0
 inactive_no_flow_5_34_0_0: flow_5_34_0_0 - 1000 edge_5_34_0_0 <= 0
 inactive_no_flow_5_38_0_0: flow_5_38_0_0 - 1000 edge_5_38_0_0 <= 0
 inactive_no_flow_26_1_0_0: flow_26_1_0_0 - 1000 edge_26_1_0_0 <= 0
 inactive_no_flow_26_3_0_0: flow_26_3_0_0 - 1000 edge_26_3_0_0 <= 0
 inactive_no_flow_26_5_0_0: flow_26_5_0_0 - 1000 edge_26_5_0_0 <= 0
 inactive_no_flow_26_27_0_0: flow_26_27_0_0 - 1000 edge_26_27_0_0 <= 0
 inactive_no_flow_26_31_0_0: flow_26_31_0_0 - 1000 edge_26_31_0_0 <= 0
 inactive_no_flow_26_33_0_0: flow_26_33_0_0 - 1000 edge_26_33_0_0 <= 0
 inactive_no_flow_26_34_0_0: flow_26_34_0_0 - 1000 edge_26_34_0_0 <= 0
 inactive_no_flow_26_38_0_0: flow_26_38_0_0 - 1000 edge_26_38_0_0 <= 0
 inactive_no_flow_27_1_0_0: flow_27_1_0_0 - 1000 edge_27_1_0_0 <= 0
 inactive_no_flow_27_3_0_0: flow_27_3_0_0 - 1000 edge_27_3_0_0 <= 0
 inactive_no_flow_27_5_0_0: flow_27_5_0_0 - 1000 edge_27_5_0_0 <= 0
 inactive_no_flow_27_26_0_0: flow_27_26_0_0 - 1000 edge_27_26_0_0 <= 0
 inactive_no_flow_27_31_0_0: flow_27_31_0_0 - 1000 edge_27_31_0_0 <= 0
 inactive_no_flow_27_33_0_0: flow_27_33_0_0 - 1000 edge_27_33_0_0 <= 0
 inactive_no_flow_27_34_0_0: flow_27_34_0_0 - 1000 edge_27_34_0_0 <= 0
 inactive_no_flow_27_38_0_0: flow_27_38_0_0 - 1000 edge_27_38_0_0 <= 0
 inactive_no_flow_31_1_0_0: flow_31_1_0_0 - 1000 edge_31_1_0_0 <= 0
 inactive_no_flow_31_3_0_0: flow_31_3_0_0 - 1000 edge_31_3_0_0 <= 0
 inactive_no_flow_31_26_0_0: flow_31_26_0_0 - 1000 edge_31_26_0_0 <= 0
 inactive_no_flow_31_27_0_0: flow_31_27_0_0 - 1000 edge_31_27_0_0 <= 0
 inactive_no_flow_31_33_0_0: flow_31_33_0_0 - 1000 edge_31_33_0_0 <= 0
 inactive_no_flow_31_34_0_0: flow_31_34_0_0 - 1000 edge_31_34_0_0 <= 0
 inactive_no_flow_31_38_0_0: flow_31_38_0_0 - 1000 edge_31_38_0_0 <= 0
 inactive_no_flow_33_1_0_0: flow_33_1_0_0 - 1000 edge_33_1_0_0 <= 0
 inactive_no_flow_33_3_0_0: flow_33_3_0_0 - 1000 edge_33_3_0_0 <= 0
 inactive_no_flow_33_5_0_0: flow_33_5_0_0 - 1000 edge_33_5_0_0 <= 0
 inactive_no_flow_33_26_0_0: flow_33_26_0_0 - 1000 edge_33_26_0_0 <= 0
 inactive_no_flow_33_27_0_0: flow_33_27_0_0 - 1000 edge_33_27_0_0 <= 0
 inactive_no_flow_33_31_0_0: flow_33_31_0_0 - 1000 edge_33_31_0_0 <= 0
 inactive_no_flow_33_34_0_0: flow_33_34_0_0 - 1000 edge_33_34_0_0 <= 0
 inactive_no_flow_33_38_0_0: flow_33_38_0_0 - 1000 edge_33_38_0_0 <= 0
 inactive_no_flow_34_1_0_0: flow_34_1_0_0 - 1000 edge_34_1_0_0 <= 0
 inactive_no_flow_34_3_0_0: flow_34_3_0_0 - 1000 edge_34_3_0_0 <= 0
 inactive_no_flow_34_5_0_0: flow_34_5_0_0 - 1000 edge_34_5_0_0 <= 0
 inactive_no_flow_34_26_0_0: flow_34_26_0_0 - 1000 edge_34_26_0_0 <= 0
 inactive_no_flow_34_27_0_0: flow_34_27_0_0 - 1000 edge_34_27_0_0 <= 0
 inactive_no_flow_34_31_0_0: flow_34_31_0_0 - 1000 edge_34_31_0_0 <= 0
 inactive_no_flow_34_33_0_0: flow_34_33_0_0 - 1000 edge_34_33_0_0 <= 0
 inactive_no_flow_34_38_0_0: flow_34_38_0_0 - 1000 edge_34_38_0_0 <= 0
 inactive_no_flow_38_31_0_0: flow_38_31_0_0 - 1000 edge_38_31_0_0 <= 0
 inactive_no_flow_38_34_0_0: flow_38_34_0_0 - 1000 edge_38_34_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_26_0_0 edge_0_27_0_0
 edge_0_31_0_0 edge_0_33_0_0 edge_0_34_0_0 edge_1_3_0_0 edge_1_5_0_0
 edge_1_26_0_0 edge_1_27_0_0 edge_1_31_0_0 edge_1_33_0_0 edge_1_34_0_0
 edge_1_38_0_0 edge_3_1_0_0 edge_3_5_0_0 edge_3_26_0_0 edge_3_27_0_0
 edge_3_31_0_0 edge_3_33_0_0 edge_3_34_0_0 edge_3_38_0_0 edge_5_1_0_0
 edge_5_3_0_0 edge_5_26_0_0 edge_5_27_0_0 edge_5_31_0_0 edge_5_33_0_0
 edge_5_34_0_0 edge_5_38_0_0 edge_26_1_0_0 edge_26_3_0_0 edge_26_5_0_0
 edge_26_27_0_0 edge_26_31_0_0 edge_26_33_0_0 edge_26_34_0_0 edge_26_38_0_0
 edge_27_1_0_0 edge_27_3_0_0 edge_27_5_0_0 edge_27_26_0_0 edge_27_31_0_0
 edge_27_33_0_0 edge_27_34_0_0 edge_27_38_0_0 edge_31_1_0_0 edge_31_3_0_0
 edge_31_26_0_0 edge_31_27_0_0 edge_31_33_0_0 edge_31_34_0_0 edge_31_38_0_0
 edge_33_1_0_0 edge_33_3_0_0 edge_33_5_0_0 edge_33_26_0_0 edge_33_27_0_0
 edge_33_31_0_0 edge_33_34_0_0 edge_33_38_0_0 edge_34_1_0_0 edge_34_3_0_0
 edge_34_5_0_0 edge_34_26_0_0 edge_34_27_0_0 edge_34_31_0_0 edge_34_33_0_0
 edge_34_38_0_0 edge_38_31_0_0 edge_38_34_0_0
End
