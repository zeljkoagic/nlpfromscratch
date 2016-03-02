Maximize
  - 4.320028380748 edge_0_1_0_0 - 1.7113514640663 edge_0_2_0_0
   - 2.536404940656 edge_0_6_0_0 - 3.5836598174786 edge_0_10_0_0
   - 1.5814858835029 edge_0_12_0_0 - 4.2228442146955 edge_0_13_0_0
   + 0.037665414898509 edge_1_2_0_0 - 0.3652122978176313 edge_1_6_0_0
   + 0.4449147481048193 edge_1_8_0_0 + 0.0208923701048824 edge_1_9_0_0
   - 0.1611433811487444 edge_1_10_0_0 + 0.1338638401969382 edge_1_11_0_0
   - 0.0855891025887576 edge_1_12_0_0 + 0.89796318630351 edge_1_13_0_0
   - 0.0542816966438669 edge_2_1_0_0 + 0.0641230745375638 edge_2_12_0_0
   + 0.4763126186475186 edge_2_13_0_0 - 0.4268244425986145 edge_6_1_0_0
   - 0.3249739310596194 edge_6_10_0_0 - 0.1336451803291185 edge_6_11_0_0
   - 0.0895412557416 edge_6_13_0_0 - 0.9139402094262534 edge_8_1_0_0
   + 0.9553214915850038 edge_8_9_0_0 + 0.588456332959211 edge_8_10_0_0
   - 0.0400579664362686 edge_8_11_0_0 + 0.0502129986862003 edge_8_12_0_0
   - 1.004720351720559 edge_9_1_0_0 + 0.1455367658467451 edge_9_8_0_0
   - 0.0320899382237243 edge_9_11_0_0 - 0.6160861481856812 edge_9_12_0_0
   - 0.7524640732554 edge_9_13_0_0 - 0.2385779729690721 edge_10_1_0_0
   + 0.1465814137033722 edge_10_6_0_0 + 1.523443480213557 edge_10_8_0_0
   + 3.279015839595966 edge_10_9_0_0 + 0.8222324665981158 edge_10_11_0_0
   + 0.3501106810708349 edge_10_12_0_0 + 0.4335281050408 edge_10_13_0_0
   - 0.8450290342799723 edge_11_1_0_0 - 0.3852958541160827 edge_11_8_0_0
   - 0.6726225200965299 edge_11_10_0_0 + 0.3270690742241995 edge_12_1_0_0
   + 0.2366212727561634 edge_12_2_0_0 - 0.3396819464545729 edge_12_6_0_0
   - 0.3687378199683127 edge_12_8_0_0 + 0.2936712267263779 edge_12_10_0_0
   + 1.377930708794861 edge_12_11_0_0 + 1.728563572134 edge_12_13_0_0
   - 1.773856638018277 edge_13_1_0_0 - 0.467887026946495 edge_13_2_0_0
   - 0.4205393404215543 edge_13_12_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_6_1_0_0 + edge_8_1_0_0
   + edge_9_1_0_0 + edge_10_1_0_0 + edge_11_1_0_0 + edge_12_1_0_0
   + edge_13_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_6_0_0
   - flow_1_8_0_0 - flow_1_9_0_0 - flow_1_10_0_0 - flow_1_11_0_0
   - flow_1_12_0_0 - flow_1_13_0_0 + flow_2_1_0_0 + flow_6_1_0_0
   + flow_8_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0 + flow_11_1_0_0
   + flow_12_1_0_0 + flow_13_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_12_2_0_0 + edge_13_2_0_0
   = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_12_0_0 - flow_2_13_0_0 + flow_12_2_0_0 + flow_13_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_10_6_0_0 + edge_12_6_0_0
   = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 - flow_6_1_0_0
   - flow_6_10_0_0 - flow_6_11_0_0 - flow_6_13_0_0 + flow_10_6_0_0
   + flow_12_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_1_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0
   + edge_12_8_0_0 = 1
 consume_one_flow_unit_8: flow_1_8_0_0 - flow_8_1_0_0 - flow_8_9_0_0
   - flow_8_10_0_0 - flow_8_11_0_0 - flow_8_12_0_0 + flow_9_8_0_0
   + flow_10_8_0_0 + flow_11_8_0_0 + flow_12_8_0_0 = 1
 one_parent_9: edge_1_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_1_9_0_0 + flow_8_9_0_0 - flow_9_1_0_0
   - flow_9_8_0_0 - flow_9_11_0_0 - flow_9_12_0_0 - flow_9_13_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_6_10_0_0
   + edge_8_10_0_0 + edge_11_10_0_0 + edge_12_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_6_10_0_0
   + flow_8_10_0_0 - flow_10_1_0_0 - flow_10_6_0_0 - flow_10_8_0_0
   - flow_10_9_0_0 - flow_10_11_0_0 - flow_10_12_0_0 - flow_10_13_0_0
   + flow_11_10_0_0 + flow_12_10_0_0 = 1
 one_parent_11: edge_1_11_0_0 + edge_6_11_0_0 + edge_8_11_0_0
   + edge_9_11_0_0 + edge_10_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_6_11_0_0 + flow_8_11_0_0
   + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_1_0_0 - flow_11_8_0_0
   - flow_11_10_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_2_12_0_0
   + edge_8_12_0_0 + edge_9_12_0_0 + edge_10_12_0_0 + edge_13_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_2_12_0_0
   + flow_8_12_0_0 + flow_9_12_0_0 + flow_10_12_0_0 - flow_12_1_0_0
   - flow_12_2_0_0 - flow_12_6_0_0 - flow_12_8_0_0 - flow_12_10_0_0
   - flow_12_11_0_0 - flow_12_13_0_0 + flow_13_12_0_0 = 1
 one_parent_13: edge_0_13_0_0 + edge_1_13_0_0 + edge_2_13_0_0
   + edge_6_13_0_0 + edge_9_13_0_0 + edge_10_13_0_0 + edge_12_13_0_0 = 1
 consume_one_flow_unit_13: flow_0_13_0_0 + flow_1_13_0_0 + flow_2_13_0_0
   + flow_6_13_0_0 + flow_9_13_0_0 + flow_10_13_0_0 + flow_12_13_0_0
   - flow_13_1_0_0 - flow_13_2_0_0 - flow_13_12_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_6_0_0 + flow_0_10_0_0
   + flow_0_12_0_0 + flow_0_13_0_0 = 13
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_0_13_0_0: flow_0_13_0_0 - 1000 edge_0_13_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_1_13_0_0: flow_1_13_0_0 - 1000 edge_1_13_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_13_0_0: flow_6_13_0_0 - 1000 edge_6_13_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_13_0_0: flow_9_13_0_0 - 1000 edge_9_13_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_10_13_0_0: flow_10_13_0_0 - 1000 edge_10_13_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_6_0_0: flow_12_6_0_0 - 1000 edge_12_6_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
 inactive_no_flow_12_13_0_0: flow_12_13_0_0 - 1000 edge_12_13_0_0 <= 0
 inactive_no_flow_13_1_0_0: flow_13_1_0_0 - 1000 edge_13_1_0_0 <= 0
 inactive_no_flow_13_2_0_0: flow_13_2_0_0 - 1000 edge_13_2_0_0 <= 0
 inactive_no_flow_13_12_0_0: flow_13_12_0_0 - 1000 edge_13_12_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_6_0_0 edge_0_10_0_0 edge_0_12_0_0
 edge_0_13_0_0 edge_1_2_0_0 edge_1_6_0_0 edge_1_8_0_0 edge_1_9_0_0
 edge_1_10_0_0 edge_1_11_0_0 edge_1_12_0_0 edge_1_13_0_0 edge_2_1_0_0
 edge_2_12_0_0 edge_2_13_0_0 edge_6_1_0_0 edge_6_10_0_0 edge_6_11_0_0
 edge_6_13_0_0 edge_8_1_0_0 edge_8_9_0_0 edge_8_10_0_0 edge_8_11_0_0
 edge_8_12_0_0 edge_9_1_0_0 edge_9_8_0_0 edge_9_11_0_0 edge_9_12_0_0
 edge_9_13_0_0 edge_10_1_0_0 edge_10_6_0_0 edge_10_8_0_0 edge_10_9_0_0
 edge_10_11_0_0 edge_10_12_0_0 edge_10_13_0_0 edge_11_1_0_0 edge_11_8_0_0
 edge_11_10_0_0 edge_12_1_0_0 edge_12_2_0_0 edge_12_6_0_0 edge_12_8_0_0
 edge_12_10_0_0 edge_12_11_0_0 edge_12_13_0_0 edge_13_1_0_0 edge_13_2_0_0
 edge_13_12_0_0
End
