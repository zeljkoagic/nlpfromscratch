Maximize
  - 4.7945105842341 edge_0_2_0_0 - 5.5289069088975 edge_0_3_0_0
   - 4.535520448041 edge_0_5_0_0 - 4.976803837079101 edge_0_8_0_0
   - 4.7668447478431 edge_0_9_0_0 - 4.6409573026684 edge_0_11_0_0
   + 1.313967793125202 edge_2_3_0_0 + 2.659952536423113 edge_2_5_0_0
   + 1.618640643792242 edge_2_6_0_0 + 2.053200999702484 edge_2_8_0_0
   + 1.369698915029604 edge_2_9_0_0 + 1.819167442355696 edge_2_11_0_0
   + 2.957086798795704 edge_2_13_0_0 + 2.571807850652815 edge_2_14_0_0
   - 1.041343188359085 edge_3_2_0_0 - 0.533449514340581 edge_3_5_0_0
   - 0.7559294689409678 edge_3_6_0_0 + 0.3191980287512501 edge_5_2_0_0
   + 1.296279756391427 edge_5_3_0_0 + 1.637670608216667 edge_5_6_0_0
   + 1.755744617921906 edge_5_8_0_0 + 1.612908807544497 edge_5_9_0_0
   + 0.93516410672823 edge_5_11_0_0 + 2.017457906072608 edge_5_13_0_0
   + 1.340013828996146 edge_5_14_0_0 - 1.326593530015246 edge_6_3_0_0
   - 0.5361591622035496 edge_6_8_0_0 - 0.7551488358753322 edge_6_9_0_0
   - 1.277431997217804 edge_6_11_0_0 - 0.645400688922998 edge_8_2_0_0
   + 0.0793368593875069 edge_8_3_0_0 + 0.3910015410657379 edge_8_5_0_0
   + 1.295707259286027 edge_8_6_0_0 + 1.630953666647762 edge_8_9_0_0
   + 1.536706934979642 edge_8_11_0_0 + 2.109176168879689 edge_8_13_0_0
   + 1.564080849850574 edge_8_14_0_0 - 1.316838904760152 edge_9_6_0_0
   - 0.9966259316278882 edge_9_8_0_0 - 0.4940995773004108 edge_9_11_0_0
   - 1.264824727013284 edge_9_14_0_0 - 1.45302137034516 edge_11_2_0_0
   - 0.8839859110239591 edge_11_3_0_0 - 0.3825018645586403 edge_11_5_0_0
   - 0.2494650543915022 edge_11_6_0_0 + 0.5730192239061281 edge_11_8_0_0
   + 0.7106381682287358 edge_11_9_0_0 + 1.924174843610666 edge_11_13_0_0
   + 1.726285062555731 edge_11_14_0_0 - 1.248188664093922 edge_13_6_0_0
   - 1.267623731557489 edge_13_8_0_0 - 0.8052544874971221 edge_13_9_0_0
   - 0.5603481953215538 edge_13_11_0_0 - 0.5063833929998952 edge_13_14_0_0
   - 1.70850657227206 edge_14_6_0_0 - 1.343263039670381 edge_14_8_0_0
   - 1.058589641039028 edge_14_9_0_0 - 0.3566363051720118 edge_14_11_0_0
   - 0.194453260230365 edge_14_13_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0 + edge_8_2_0_0
   + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_5_0_0
   - flow_2_6_0_0 - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_11_0_0
   - flow_2_13_0_0 - flow_2_14_0_0 + flow_3_2_0_0 + flow_5_2_0_0
   + flow_8_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0
   + edge_8_3_0_0 + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_5_0_0 - flow_3_6_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_8_3_0_0 + flow_11_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_8_5_0_0
   + edge_11_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_6_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 - flow_5_11_0_0 - flow_5_13_0_0 - flow_5_14_0_0
   + flow_8_5_0_0 + flow_11_5_0_0 = 1
 one_parent_6: edge_2_6_0_0 + edge_3_6_0_0 + edge_5_6_0_0 + edge_8_6_0_0
   + edge_9_6_0_0 + edge_11_6_0_0 + edge_13_6_0_0 + edge_14_6_0_0 = 1
 consume_one_flow_unit_6: flow_2_6_0_0 + flow_3_6_0_0 + flow_5_6_0_0
   - flow_6_3_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_11_0_0
   + flow_8_6_0_0 + flow_9_6_0_0 + flow_11_6_0_0 + flow_13_6_0_0
   + flow_14_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0
   + edge_9_8_0_0 + edge_11_8_0_0 + edge_13_8_0_0 + edge_14_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_5_8_0_0
   + flow_6_8_0_0 - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_9_0_0 - flow_8_11_0_0 - flow_8_13_0_0
   - flow_8_14_0_0 + flow_9_8_0_0 + flow_11_8_0_0 + flow_13_8_0_0
   + flow_14_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0
   + edge_8_9_0_0 + edge_11_9_0_0 + edge_13_9_0_0 + edge_14_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_5_9_0_0
   + flow_6_9_0_0 + flow_8_9_0_0 - flow_9_6_0_0 - flow_9_8_0_0
   - flow_9_11_0_0 - flow_9_14_0_0 + flow_11_9_0_0 + flow_13_9_0_0
   + flow_14_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_0_11_0_0 + edge_2_11_0_0 + edge_5_11_0_0
   + edge_6_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0 + edge_13_11_0_0
   + edge_14_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_2_11_0_0 + flow_5_11_0_0
   + flow_6_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0 - flow_11_2_0_0
   - flow_11_3_0_0 - flow_11_5_0_0 - flow_11_6_0_0 - flow_11_8_0_0
   - flow_11_9_0_0 - flow_11_13_0_0 - flow_11_14_0_0 + flow_13_11_0_0
   + flow_14_11_0_0 = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: edge_2_13_0_0 + edge_5_13_0_0 + edge_8_13_0_0
   + edge_11_13_0_0 + edge_14_13_0_0 = 1
 consume_one_flow_unit_13: flow_2_13_0_0 + flow_5_13_0_0 + flow_8_13_0_0
   + flow_11_13_0_0 - flow_13_6_0_0 - flow_13_8_0_0 - flow_13_9_0_0
   - flow_13_11_0_0 - flow_13_14_0_0 + flow_14_13_0_0 = 1
 one_parent_14: edge_2_14_0_0 + edge_5_14_0_0 + edge_8_14_0_0
   + edge_9_14_0_0 + edge_11_14_0_0 + edge_13_14_0_0 = 1
 consume_one_flow_unit_14: flow_2_14_0_0 + flow_5_14_0_0 + flow_8_14_0_0
   + flow_9_14_0_0 + flow_11_14_0_0 + flow_13_14_0_0 - flow_14_6_0_0
   - flow_14_8_0_0 - flow_14_9_0_0 - flow_14_11_0_0 - flow_14_13_0_0 = 1
 one_parent_15: = 1
 consume_one_flow_unit_15: = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_8_0_0
   + flow_0_9_0_0 + flow_0_11_0_0 = 15
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_2_14_0_0: flow_2_14_0_0 - 1000 edge_2_14_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_13_0_0: flow_5_13_0_0 - 1000 edge_5_13_0_0 <= 0
 inactive_no_flow_5_14_0_0: flow_5_14_0_0 - 1000 edge_5_14_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_13_0_0: flow_8_13_0_0 - 1000 edge_8_13_0_0 <= 0
 inactive_no_flow_8_14_0_0: flow_8_14_0_0 - 1000 edge_8_14_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_14_0_0: flow_9_14_0_0 - 1000 edge_9_14_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_13_0_0: flow_11_13_0_0 - 1000 edge_11_13_0_0 <= 0
 inactive_no_flow_11_14_0_0: flow_11_14_0_0 - 1000 edge_11_14_0_0 <= 0
 inactive_no_flow_13_6_0_0: flow_13_6_0_0 - 1000 edge_13_6_0_0 <= 0
 inactive_no_flow_13_8_0_0: flow_13_8_0_0 - 1000 edge_13_8_0_0 <= 0
 inactive_no_flow_13_9_0_0: flow_13_9_0_0 - 1000 edge_13_9_0_0 <= 0
 inactive_no_flow_13_11_0_0: flow_13_11_0_0 - 1000 edge_13_11_0_0 <= 0
 inactive_no_flow_13_14_0_0: flow_13_14_0_0 - 1000 edge_13_14_0_0 <= 0
 inactive_no_flow_14_6_0_0: flow_14_6_0_0 - 1000 edge_14_6_0_0 <= 0
 inactive_no_flow_14_8_0_0: flow_14_8_0_0 - 1000 edge_14_8_0_0 <= 0
 inactive_no_flow_14_9_0_0: flow_14_9_0_0 - 1000 edge_14_9_0_0 <= 0
 inactive_no_flow_14_11_0_0: flow_14_11_0_0 - 1000 edge_14_11_0_0 <= 0
 inactive_no_flow_14_13_0_0: flow_14_13_0_0 - 1000 edge_14_13_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_8_0_0 edge_0_9_0_0
 edge_0_11_0_0 edge_2_3_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_8_0_0
 edge_2_9_0_0 edge_2_11_0_0 edge_2_13_0_0 edge_2_14_0_0 edge_3_2_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_6_0_0
 edge_5_8_0_0 edge_5_9_0_0 edge_5_11_0_0 edge_5_13_0_0 edge_5_14_0_0
 edge_6_3_0_0 edge_6_8_0_0 edge_6_9_0_0 edge_6_11_0_0 edge_8_2_0_0
 edge_8_3_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_9_0_0 edge_8_11_0_0
 edge_8_13_0_0 edge_8_14_0_0 edge_9_6_0_0 edge_9_8_0_0 edge_9_11_0_0
 edge_9_14_0_0 edge_11_2_0_0 edge_11_3_0_0 edge_11_5_0_0 edge_11_6_0_0
 edge_11_8_0_0 edge_11_9_0_0 edge_11_13_0_0 edge_11_14_0_0 edge_13_6_0_0
 edge_13_8_0_0 edge_13_9_0_0 edge_13_11_0_0 edge_13_14_0_0 edge_14_6_0_0
 edge_14_8_0_0 edge_14_9_0_0 edge_14_11_0_0 edge_14_13_0_0
End
