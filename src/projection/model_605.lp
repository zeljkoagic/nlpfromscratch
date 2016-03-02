Maximize
  - 1.7176456336368 edge_0_1_0_0 + 0.31583515442963 edge_0_7_0_0
   + 0.07477389071715 edge_0_16_0_0 - 0.5846306076809999 edge_0_23_0_0
   - 0.0935627483319149 edge_1_2_0_0 + 0.0622677634819002 edge_1_5_0_0
   - 0.1054246151912854 edge_1_7_0_0 + 0.2715228161689421 edge_1_16_0_0
   + 0.7128804535312829 edge_1_40_0_0 - 0.0107386295706861 edge_2_1_0_0
   + 0.1858566490741345 edge_2_5_0_0 + 0.1078866412528712 edge_2_7_0_0
   - 0.0142065091368262 edge_2_16_0_0 + 0.0764946219830943 edge_2_21_0_0
   + 0.0197980509641659 edge_2_23_0_0 - 0.0802593208727622 edge_2_38_0_0
   + 0.2763227072460955 edge_2_40_0_0 - 0.3797612000429758 edge_5_1_0_0
   + 0.0211557018003727 edge_5_2_0_0 + 0.00909607249874672 edge_5_7_0_0
   - 0.2969678504798645 edge_5_16_0_0 - 0.1447387408529502 edge_5_21_0_0
   - 0.0985915656731878 edge_5_23_0_0 - 0.2932446333091746 edge_5_38_0_0
   + 0.0558415194384548 edge_7_1_0_0 + 0.2389184563726032 edge_7_2_0_0
   + 0.4343099148682045 edge_7_5_0_0 + 0.4141776418458056 edge_7_16_0_0
   + 1.117991757321869 edge_7_21_0_0 + 0.7751989133595908 edge_7_23_0_0
   + 1.035432337467342 edge_7_38_0_0 + 1.859781384366304 edge_7_40_0_0
   + 0.6060826551390194 edge_16_1_0_0 + 0.4343091248424429 edge_16_2_0_0
   + 0.6293348938710959 edge_16_5_0_0 + 0.521086738613844 edge_16_7_0_0
   + 0.3035251060939287 edge_16_21_0_0 + 0.3132323375507094 edge_16_23_0_0
   + 0.2012701048127943 edge_16_38_0_0 + 1.544259615385826 edge_16_40_0_0
   + 0.2078559296075464 edge_21_23_0_0 + 0.1721951202993983 edge_21_38_0_0
   + 0.2341206086206192 edge_21_40_0_0 - 0.0914079308238676 edge_23_1_0_0
   + 0.0170639542823984 edge_23_2_0_0 + 0.1199070057561397 edge_23_5_0_0
   + 0.2381857527274464 edge_23_7_0_0 + 0.1409767445201339 edge_23_16_0_0
   + 0.6932328166626708 edge_23_21_0_0 + 1.287825280156498 edge_23_38_0_0
   + 1.112098561144649 edge_23_40_0_0 - 0.5840290199869467 edge_38_1_0_0
   - 0.2566549043337636 edge_38_2_0_0 - 0.0300854123900439 edge_38_5_0_0
   - 0.00171797226402142 edge_38_16_0_0 + 0.2926084472119212 edge_38_21_0_0
   + 0.3124548459572433 edge_38_23_0_0 + 1.204122405119397 edge_38_40_0_0
   - 1.022483488501753 edge_40_1_0_0 - 0.512307310150232 edge_40_2_0_0
   - 0.5481701730343731 edge_40_5_0_0 - 0.6189733858356016 edge_40_7_0_0
   - 0.3485366106143029 edge_40_21_0_0 - 0.4404746601616505 edge_40_23_0_0
   - 0.4571365686950122 edge_40_38_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_5_1_0_0 + edge_7_1_0_0
   + edge_16_1_0_0 + edge_23_1_0_0 + edge_38_1_0_0 + edge_40_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_5_0_0
   - flow_1_7_0_0 - flow_1_16_0_0 - flow_1_40_0_0 + flow_2_1_0_0
   + flow_5_1_0_0 + flow_7_1_0_0 + flow_16_1_0_0 + flow_23_1_0_0
   + flow_38_1_0_0 + flow_40_1_0_0 = 1
 one_parent_2: edge_1_2_0_0 + edge_5_2_0_0 + edge_7_2_0_0 + edge_16_2_0_0
   + edge_23_2_0_0 + edge_38_2_0_0 + edge_40_2_0_0 = 1
 consume_one_flow_unit_2: flow_1_2_0_0 - flow_2_1_0_0 - flow_2_5_0_0
   - flow_2_7_0_0 - flow_2_16_0_0 - flow_2_21_0_0 - flow_2_23_0_0
   - flow_2_38_0_0 - flow_2_40_0_0 + flow_5_2_0_0 + flow_7_2_0_0
   + flow_16_2_0_0 + flow_23_2_0_0 + flow_38_2_0_0 + flow_40_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_1_5_0_0 + edge_2_5_0_0 + edge_7_5_0_0 + edge_16_5_0_0
   + edge_23_5_0_0 + edge_38_5_0_0 + edge_40_5_0_0 = 1
 consume_one_flow_unit_5: flow_1_5_0_0 + flow_2_5_0_0 - flow_5_1_0_0
   - flow_5_2_0_0 - flow_5_7_0_0 - flow_5_16_0_0 - flow_5_21_0_0
   - flow_5_23_0_0 - flow_5_38_0_0 + flow_7_5_0_0 + flow_16_5_0_0
   + flow_23_5_0_0 + flow_38_5_0_0 + flow_40_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_5_7_0_0
   + edge_16_7_0_0 + edge_23_7_0_0 + edge_40_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_5_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_5_0_0
   - flow_7_16_0_0 - flow_7_21_0_0 - flow_7_23_0_0 - flow_7_38_0_0
   - flow_7_40_0_0 + flow_16_7_0_0 + flow_23_7_0_0 + flow_40_7_0_0 = 1
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
 one_parent_16: edge_0_16_0_0 + edge_1_16_0_0 + edge_2_16_0_0
   + edge_5_16_0_0 + edge_7_16_0_0 + edge_23_16_0_0 + edge_38_16_0_0 = 1
 consume_one_flow_unit_16: flow_0_16_0_0 + flow_1_16_0_0 + flow_2_16_0_0
   + flow_5_16_0_0 + flow_7_16_0_0 - flow_16_1_0_0 - flow_16_2_0_0
   - flow_16_5_0_0 - flow_16_7_0_0 - flow_16_21_0_0 - flow_16_23_0_0
   - flow_16_38_0_0 - flow_16_40_0_0 + flow_23_16_0_0 + flow_38_16_0_0 = 1
 one_parent_17: = 1
 consume_one_flow_unit_17: = 1
 one_parent_18: = 1
 consume_one_flow_unit_18: = 1
 one_parent_19: = 1
 consume_one_flow_unit_19: = 1
 one_parent_20: = 1
 consume_one_flow_unit_20: = 1
 one_parent_21: edge_2_21_0_0 + edge_5_21_0_0 + edge_7_21_0_0
   + edge_16_21_0_0 + edge_23_21_0_0 + edge_38_21_0_0 + edge_40_21_0_0 = 1
 consume_one_flow_unit_21: flow_2_21_0_0 + flow_5_21_0_0 + flow_7_21_0_0
   + flow_16_21_0_0 - flow_21_23_0_0 - flow_21_38_0_0 - flow_21_40_0_0
   + flow_23_21_0_0 + flow_38_21_0_0 + flow_40_21_0_0 = 1
 one_parent_22: = 1
 consume_one_flow_unit_22: = 1
 one_parent_23: edge_0_23_0_0 + edge_2_23_0_0 + edge_5_23_0_0
   + edge_7_23_0_0 + edge_16_23_0_0 + edge_21_23_0_0 + edge_38_23_0_0
   + edge_40_23_0_0 = 1
 consume_one_flow_unit_23: flow_0_23_0_0 + flow_2_23_0_0 + flow_5_23_0_0
   + flow_7_23_0_0 + flow_16_23_0_0 + flow_21_23_0_0 - flow_23_1_0_0
   - flow_23_2_0_0 - flow_23_5_0_0 - flow_23_7_0_0 - flow_23_16_0_0
   - flow_23_21_0_0 - flow_23_38_0_0 - flow_23_40_0_0 + flow_38_23_0_0
   + flow_40_23_0_0 = 1
 one_parent_24: = 1
 consume_one_flow_unit_24: = 1
 one_parent_25: = 1
 consume_one_flow_unit_25: = 1
 one_parent_26: = 1
 consume_one_flow_unit_26: = 1
 one_parent_27: = 1
 consume_one_flow_unit_27: = 1
 one_parent_28: = 1
 consume_one_flow_unit_28: = 1
 one_parent_29: = 1
 consume_one_flow_unit_29: = 1
 one_parent_30: = 1
 consume_one_flow_unit_30: = 1
 one_parent_31: = 1
 consume_one_flow_unit_31: = 1
 one_parent_32: = 1
 consume_one_flow_unit_32: = 1
 one_parent_33: = 1
 consume_one_flow_unit_33: = 1
 one_parent_34: = 1
 consume_one_flow_unit_34: = 1
 one_parent_35: = 1
 consume_one_flow_unit_35: = 1
 one_parent_36: = 1
 consume_one_flow_unit_36: = 1
 one_parent_37: = 1
 consume_one_flow_unit_37: = 1
 one_parent_38: edge_2_38_0_0 + edge_5_38_0_0 + edge_7_38_0_0
   + edge_16_38_0_0 + edge_21_38_0_0 + edge_23_38_0_0 + edge_40_38_0_0 = 1
 consume_one_flow_unit_38: flow_2_38_0_0 + flow_5_38_0_0 + flow_7_38_0_0
   + flow_16_38_0_0 + flow_21_38_0_0 + flow_23_38_0_0 - flow_38_1_0_0
   - flow_38_2_0_0 - flow_38_5_0_0 - flow_38_16_0_0 - flow_38_21_0_0
   - flow_38_23_0_0 - flow_38_40_0_0 + flow_40_38_0_0 = 1
 one_parent_39: = 1
 consume_one_flow_unit_39: = 1
 one_parent_40: edge_1_40_0_0 + edge_2_40_0_0 + edge_7_40_0_0
   + edge_16_40_0_0 + edge_21_40_0_0 + edge_23_40_0_0 + edge_38_40_0_0 = 1
 consume_one_flow_unit_40: flow_1_40_0_0 + flow_2_40_0_0 + flow_7_40_0_0
   + flow_16_40_0_0 + flow_21_40_0_0 + flow_23_40_0_0 + flow_38_40_0_0
   - flow_40_1_0_0 - flow_40_2_0_0 - flow_40_5_0_0 - flow_40_7_0_0
   - flow_40_21_0_0 - flow_40_23_0_0 - flow_40_38_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_7_0_0 + flow_0_16_0_0 + flow_0_23_0_0
   = 40
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_16_0_0: flow_0_16_0_0 - 1000 edge_0_16_0_0 <= 0
 inactive_no_flow_0_23_0_0: flow_0_23_0_0 - 1000 edge_0_23_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_16_0_0: flow_1_16_0_0 - 1000 edge_1_16_0_0 <= 0
 inactive_no_flow_1_40_0_0: flow_1_40_0_0 - 1000 edge_1_40_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_16_0_0: flow_2_16_0_0 - 1000 edge_2_16_0_0 <= 0
 inactive_no_flow_2_21_0_0: flow_2_21_0_0 - 1000 edge_2_21_0_0 <= 0
 inactive_no_flow_2_23_0_0: flow_2_23_0_0 - 1000 edge_2_23_0_0 <= 0
 inactive_no_flow_2_38_0_0: flow_2_38_0_0 - 1000 edge_2_38_0_0 <= 0
 inactive_no_flow_2_40_0_0: flow_2_40_0_0 - 1000 edge_2_40_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_16_0_0: flow_5_16_0_0 - 1000 edge_5_16_0_0 <= 0
 inactive_no_flow_5_21_0_0: flow_5_21_0_0 - 1000 edge_5_21_0_0 <= 0
 inactive_no_flow_5_23_0_0: flow_5_23_0_0 - 1000 edge_5_23_0_0 <= 0
 inactive_no_flow_5_38_0_0: flow_5_38_0_0 - 1000 edge_5_38_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_16_0_0: flow_7_16_0_0 - 1000 edge_7_16_0_0 <= 0
 inactive_no_flow_7_21_0_0: flow_7_21_0_0 - 1000 edge_7_21_0_0 <= 0
 inactive_no_flow_7_23_0_0: flow_7_23_0_0 - 1000 edge_7_23_0_0 <= 0
 inactive_no_flow_7_38_0_0: flow_7_38_0_0 - 1000 edge_7_38_0_0 <= 0
 inactive_no_flow_7_40_0_0: flow_7_40_0_0 - 1000 edge_7_40_0_0 <= 0
 inactive_no_flow_16_1_0_0: flow_16_1_0_0 - 1000 edge_16_1_0_0 <= 0
 inactive_no_flow_16_2_0_0: flow_16_2_0_0 - 1000 edge_16_2_0_0 <= 0
 inactive_no_flow_16_5_0_0: flow_16_5_0_0 - 1000 edge_16_5_0_0 <= 0
 inactive_no_flow_16_7_0_0: flow_16_7_0_0 - 1000 edge_16_7_0_0 <= 0
 inactive_no_flow_16_21_0_0: flow_16_21_0_0 - 1000 edge_16_21_0_0 <= 0
 inactive_no_flow_16_23_0_0: flow_16_23_0_0 - 1000 edge_16_23_0_0 <= 0
 inactive_no_flow_16_38_0_0: flow_16_38_0_0 - 1000 edge_16_38_0_0 <= 0
 inactive_no_flow_16_40_0_0: flow_16_40_0_0 - 1000 edge_16_40_0_0 <= 0
 inactive_no_flow_21_23_0_0: flow_21_23_0_0 - 1000 edge_21_23_0_0 <= 0
 inactive_no_flow_21_38_0_0: flow_21_38_0_0 - 1000 edge_21_38_0_0 <= 0
 inactive_no_flow_21_40_0_0: flow_21_40_0_0 - 1000 edge_21_40_0_0 <= 0
 inactive_no_flow_23_1_0_0: flow_23_1_0_0 - 1000 edge_23_1_0_0 <= 0
 inactive_no_flow_23_2_0_0: flow_23_2_0_0 - 1000 edge_23_2_0_0 <= 0
 inactive_no_flow_23_5_0_0: flow_23_5_0_0 - 1000 edge_23_5_0_0 <= 0
 inactive_no_flow_23_7_0_0: flow_23_7_0_0 - 1000 edge_23_7_0_0 <= 0
 inactive_no_flow_23_16_0_0: flow_23_16_0_0 - 1000 edge_23_16_0_0 <= 0
 inactive_no_flow_23_21_0_0: flow_23_21_0_0 - 1000 edge_23_21_0_0 <= 0
 inactive_no_flow_23_38_0_0: flow_23_38_0_0 - 1000 edge_23_38_0_0 <= 0
 inactive_no_flow_23_40_0_0: flow_23_40_0_0 - 1000 edge_23_40_0_0 <= 0
 inactive_no_flow_38_1_0_0: flow_38_1_0_0 - 1000 edge_38_1_0_0 <= 0
 inactive_no_flow_38_2_0_0: flow_38_2_0_0 - 1000 edge_38_2_0_0 <= 0
 inactive_no_flow_38_5_0_0: flow_38_5_0_0 - 1000 edge_38_5_0_0 <= 0
 inactive_no_flow_38_16_0_0: flow_38_16_0_0 - 1000 edge_38_16_0_0 <= 0
 inactive_no_flow_38_21_0_0: flow_38_21_0_0 - 1000 edge_38_21_0_0 <= 0
 inactive_no_flow_38_23_0_0: flow_38_23_0_0 - 1000 edge_38_23_0_0 <= 0
 inactive_no_flow_38_40_0_0: flow_38_40_0_0 - 1000 edge_38_40_0_0 <= 0
 inactive_no_flow_40_1_0_0: flow_40_1_0_0 - 1000 edge_40_1_0_0 <= 0
 inactive_no_flow_40_2_0_0: flow_40_2_0_0 - 1000 edge_40_2_0_0 <= 0
 inactive_no_flow_40_5_0_0: flow_40_5_0_0 - 1000 edge_40_5_0_0 <= 0
 inactive_no_flow_40_7_0_0: flow_40_7_0_0 - 1000 edge_40_7_0_0 <= 0
 inactive_no_flow_40_21_0_0: flow_40_21_0_0 - 1000 edge_40_21_0_0 <= 0
 inactive_no_flow_40_23_0_0: flow_40_23_0_0 - 1000 edge_40_23_0_0 <= 0
 inactive_no_flow_40_38_0_0: flow_40_38_0_0 - 1000 edge_40_38_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_7_0_0 edge_0_16_0_0 edge_0_23_0_0 edge_1_2_0_0
 edge_1_5_0_0 edge_1_7_0_0 edge_1_16_0_0 edge_1_40_0_0 edge_2_1_0_0
 edge_2_5_0_0 edge_2_7_0_0 edge_2_16_0_0 edge_2_21_0_0 edge_2_23_0_0
 edge_2_38_0_0 edge_2_40_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_7_0_0
 edge_5_16_0_0 edge_5_21_0_0 edge_5_23_0_0 edge_5_38_0_0 edge_7_1_0_0
 edge_7_2_0_0 edge_7_5_0_0 edge_7_16_0_0 edge_7_21_0_0 edge_7_23_0_0
 edge_7_38_0_0 edge_7_40_0_0 edge_16_1_0_0 edge_16_2_0_0 edge_16_5_0_0
 edge_16_7_0_0 edge_16_21_0_0 edge_16_23_0_0 edge_16_38_0_0 edge_16_40_0_0
 edge_21_23_0_0 edge_21_38_0_0 edge_21_40_0_0 edge_23_1_0_0 edge_23_2_0_0
 edge_23_5_0_0 edge_23_7_0_0 edge_23_16_0_0 edge_23_21_0_0 edge_23_38_0_0
 edge_23_40_0_0 edge_38_1_0_0 edge_38_2_0_0 edge_38_5_0_0 edge_38_16_0_0
 edge_38_21_0_0 edge_38_23_0_0 edge_38_40_0_0 edge_40_1_0_0 edge_40_2_0_0
 edge_40_5_0_0 edge_40_7_0_0 edge_40_21_0_0 edge_40_23_0_0 edge_40_38_0_0
End
