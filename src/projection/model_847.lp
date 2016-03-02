Maximize
  - 4.453397317824 edge_0_23_0_0 - 2.0997568408011 edge_0_26_0_0
   - 4.7583626245265 edge_0_27_0_0 - 3.4640726535081 edge_0_29_0_0
   - 5.4005232493375 edge_0_34_0_0 + 0.315452925552698 edge_23_26_0_0
   - 0.9925952790717326 edge_23_27_0_0 - 0.3240693217301011 edge_23_29_0_0
   + 1.033860351864945 edge_23_34_0_0 + 0.811999122552751 edge_26_23_0_0
   + 0.4113606508415996 edge_26_26_0_0 + 0.6843924482945609 edge_26_27_0_0
   + 0.1445856531381901 edge_26_28_0_0 + 0.4931662862555269 edge_26_29_0_0
   + 0.8475528928719596 edge_26_34_0_0 - 0.4742332199997696 edge_27_23_0_0
   + 0.1661065811943701 edge_27_26_0_0 + 0.8763618749631684 edge_27_28_0_0
   + 0.6991182199006195 edge_27_29_0_0 + 0.7043110279249646 edge_27_34_0_0
   - 0.9956572446175684 edge_28_23_0_0 - 0.4253614635965073 edge_28_26_0_0
   - 0.7280150662528984 edge_28_27_0_0 + 0.0627467041342048 edge_28_34_0_0
   + 0.5500848106268914 edge_29_23_0_0 + 0.3171344614636409 edge_29_26_0_0
   + 1.961999277724806 edge_29_27_0_0 + 3.574644249008512 edge_29_28_0_0
   + 3.294597913199861 edge_29_34_0_0 - 0.5856047985169068 edge_34_29_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
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
 one_parent_23: edge_0_23_0_0 + edge_26_23_0_0 + edge_27_23_0_0
   + edge_28_23_0_0 + edge_29_23_0_0 = 1
 consume_one_flow_unit_23: flow_0_23_0_0 - flow_23_26_0_0 - flow_23_27_0_0
   - flow_23_29_0_0 - flow_23_34_0_0 + flow_26_23_0_0 + flow_27_23_0_0
   + flow_28_23_0_0 + flow_29_23_0_0 = 1
 one_parent_24: = 1
 consume_one_flow_unit_24: = 1
 one_parent_25: = 1
 consume_one_flow_unit_25: = 1
 one_parent_26: edge_0_26_0_0 + edge_23_26_0_0 + edge_26_26_0_0
   + edge_27_26_0_0 + edge_28_26_0_0 + edge_29_26_0_0 = 1
 consume_one_flow_unit_26: flow_0_26_0_0 + flow_23_26_0_0 - flow_26_23_0_0
   - flow_26_27_0_0 - flow_26_28_0_0 - flow_26_29_0_0 - flow_26_34_0_0
   + flow_27_26_0_0 + flow_28_26_0_0 + flow_29_26_0_0 = 1
 one_parent_27: edge_0_27_0_0 + edge_23_27_0_0 + edge_26_27_0_0
   + edge_28_27_0_0 + edge_29_27_0_0 = 1
 consume_one_flow_unit_27: flow_0_27_0_0 + flow_23_27_0_0 + flow_26_27_0_0
   - flow_27_23_0_0 - flow_27_26_0_0 - flow_27_28_0_0 - flow_27_29_0_0
   - flow_27_34_0_0 + flow_28_27_0_0 + flow_29_27_0_0 = 1
 one_parent_28: edge_26_28_0_0 + edge_27_28_0_0 + edge_29_28_0_0 = 1
 consume_one_flow_unit_28: flow_26_28_0_0 + flow_27_28_0_0 - flow_28_23_0_0
   - flow_28_26_0_0 - flow_28_27_0_0 - flow_28_34_0_0 + flow_29_28_0_0 = 1
 one_parent_29: edge_0_29_0_0 + edge_23_29_0_0 + edge_26_29_0_0
   + edge_27_29_0_0 + edge_34_29_0_0 = 1
 consume_one_flow_unit_29: flow_0_29_0_0 + flow_23_29_0_0 + flow_26_29_0_0
   + flow_27_29_0_0 - flow_29_23_0_0 - flow_29_26_0_0 - flow_29_27_0_0
   - flow_29_28_0_0 - flow_29_34_0_0 + flow_34_29_0_0 = 1
 one_parent_30: = 1
 consume_one_flow_unit_30: = 1
 one_parent_31: = 1
 consume_one_flow_unit_31: = 1
 one_parent_32: = 1
 consume_one_flow_unit_32: = 1
 one_parent_33: = 1
 consume_one_flow_unit_33: = 1
 one_parent_34: edge_0_34_0_0 + edge_23_34_0_0 + edge_26_34_0_0
   + edge_27_34_0_0 + edge_28_34_0_0 + edge_29_34_0_0 = 1
 consume_one_flow_unit_34: flow_0_34_0_0 + flow_23_34_0_0 + flow_26_34_0_0
   + flow_27_34_0_0 + flow_28_34_0_0 + flow_29_34_0_0 - flow_34_29_0_0 = 1
 root_flow: flow_0_23_0_0 + flow_0_26_0_0 + flow_0_27_0_0 + flow_0_29_0_0
   + flow_0_34_0_0 = 34
 inactive_no_flow_0_23_0_0: flow_0_23_0_0 - 1000 edge_0_23_0_0 <= 0
 inactive_no_flow_0_26_0_0: flow_0_26_0_0 - 1000 edge_0_26_0_0 <= 0
 inactive_no_flow_0_27_0_0: flow_0_27_0_0 - 1000 edge_0_27_0_0 <= 0
 inactive_no_flow_0_29_0_0: flow_0_29_0_0 - 1000 edge_0_29_0_0 <= 0
 inactive_no_flow_0_34_0_0: flow_0_34_0_0 - 1000 edge_0_34_0_0 <= 0
 inactive_no_flow_23_26_0_0: flow_23_26_0_0 - 1000 edge_23_26_0_0 <= 0
 inactive_no_flow_23_27_0_0: flow_23_27_0_0 - 1000 edge_23_27_0_0 <= 0
 inactive_no_flow_23_29_0_0: flow_23_29_0_0 - 1000 edge_23_29_0_0 <= 0
 inactive_no_flow_23_34_0_0: flow_23_34_0_0 - 1000 edge_23_34_0_0 <= 0
 inactive_no_flow_26_23_0_0: flow_26_23_0_0 - 1000 edge_26_23_0_0 <= 0
 inactive_no_flow_26_26_0_0: flow_26_26_0_0 - 1000 edge_26_26_0_0 <= 0
 inactive_no_flow_26_27_0_0: flow_26_27_0_0 - 1000 edge_26_27_0_0 <= 0
 inactive_no_flow_26_28_0_0: flow_26_28_0_0 - 1000 edge_26_28_0_0 <= 0
 inactive_no_flow_26_29_0_0: flow_26_29_0_0 - 1000 edge_26_29_0_0 <= 0
 inactive_no_flow_26_34_0_0: flow_26_34_0_0 - 1000 edge_26_34_0_0 <= 0
 inactive_no_flow_27_23_0_0: flow_27_23_0_0 - 1000 edge_27_23_0_0 <= 0
 inactive_no_flow_27_26_0_0: flow_27_26_0_0 - 1000 edge_27_26_0_0 <= 0
 inactive_no_flow_27_28_0_0: flow_27_28_0_0 - 1000 edge_27_28_0_0 <= 0
 inactive_no_flow_27_29_0_0: flow_27_29_0_0 - 1000 edge_27_29_0_0 <= 0
 inactive_no_flow_27_34_0_0: flow_27_34_0_0 - 1000 edge_27_34_0_0 <= 0
 inactive_no_flow_28_23_0_0: flow_28_23_0_0 - 1000 edge_28_23_0_0 <= 0
 inactive_no_flow_28_26_0_0: flow_28_26_0_0 - 1000 edge_28_26_0_0 <= 0
 inactive_no_flow_28_27_0_0: flow_28_27_0_0 - 1000 edge_28_27_0_0 <= 0
 inactive_no_flow_28_34_0_0: flow_28_34_0_0 - 1000 edge_28_34_0_0 <= 0
 inactive_no_flow_29_23_0_0: flow_29_23_0_0 - 1000 edge_29_23_0_0 <= 0
 inactive_no_flow_29_26_0_0: flow_29_26_0_0 - 1000 edge_29_26_0_0 <= 0
 inactive_no_flow_29_27_0_0: flow_29_27_0_0 - 1000 edge_29_27_0_0 <= 0
 inactive_no_flow_29_28_0_0: flow_29_28_0_0 - 1000 edge_29_28_0_0 <= 0
 inactive_no_flow_29_34_0_0: flow_29_34_0_0 - 1000 edge_29_34_0_0 <= 0
 inactive_no_flow_34_29_0_0: flow_34_29_0_0 - 1000 edge_34_29_0_0 <= 0
Bounds
Binaries
 edge_0_23_0_0 edge_0_26_0_0 edge_0_27_0_0 edge_0_29_0_0 edge_0_34_0_0
 edge_23_26_0_0 edge_23_27_0_0 edge_23_29_0_0 edge_23_34_0_0 edge_26_23_0_0
 edge_26_26_0_0 edge_26_27_0_0 edge_26_28_0_0 edge_26_29_0_0 edge_26_34_0_0
 edge_27_23_0_0 edge_27_26_0_0 edge_27_28_0_0 edge_27_29_0_0 edge_27_34_0_0
 edge_28_23_0_0 edge_28_26_0_0 edge_28_27_0_0 edge_28_34_0_0 edge_29_23_0_0
 edge_29_26_0_0 edge_29_27_0_0 edge_29_28_0_0 edge_29_34_0_0 edge_34_29_0_0
End
