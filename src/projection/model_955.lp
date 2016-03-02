Maximize
  - 4.53228 edge_0_2_0_0 - 2.5953085546388 edge_0_3_0_0
   - 4.8023297300741 edge_0_4_0_0 - 3.358651152792 edge_0_6_0_0
   - 2.9316538510488 edge_0_8_0_0 - 3.223941983981 edge_0_10_0_0
   - 3.5868781735418 edge_0_11_0_0 + 1.2597744443702 edge_2_3_0_0
   + 0.58393046618672 edge_2_4_0_0 + 0.79703063814972 edge_2_6_0_0
   + 0.15454965461244 edge_2_7_0_0 + 0.24638886548604 edge_2_8_0_0
   - 0.6185730512161201 edge_2_9_0_0 + 0.1340653399163 edge_2_10_0_0
   + 1.65525 edge_2_19_0_0 + 2.1514505120614 edge_3_2_0_0
   + 2.11568670614418 edge_3_4_0_0 + 0.5114881766332936 edge_3_6_0_0
   + 0.1530349498162951 edge_3_7_0_0 + 0.7839649323533617 edge_3_8_0_0
   - 0.7261220336942297 edge_3_9_0_0 + 0.0785599087957544 edge_3_10_0_0
   - 0.2455627898199712 edge_3_11_0_0 + 4.3903106386447 edge_3_19_0_0
   - 0.9421176281593201 edge_4_2_0_0 - 0.5805232182043876 edge_4_3_0_0
   - 0.6323087957172527 edge_4_6_0_0 - 0.6210511374003017 edge_4_7_0_0
   - 1.417053751295873 edge_4_8_0_0 - 0.3141498438546 edge_6_2_0_0
   - 0.2908846101415121 edge_6_3_0_0 + 0.1100077674565944 edge_6_4_0_0
   + 0.4892927383835775 edge_6_7_0_0 + 0.42741936493442 edge_6_8_0_0
   - 0.7224972042061588 edge_6_9_0_0 + 0.336326022958246 edge_6_10_0_0
   - 0.4098175808499105 edge_6_11_0_0 + 0.4964593796799 edge_6_19_0_0
   - 0.16549415805066 edge_7_2_0_0 + 0.0913809660394226 edge_7_3_0_0
   + 0.3381313878765033 edge_7_4_0_0 + 0.4374488892977639 edge_7_6_0_0
   + 0.1072679599287318 edge_7_9_0_0 - 0.1610701292772251 edge_7_11_0_0
   + 0.6309675718557399 edge_7_19_0_0 + 0.48324705134472 edge_8_2_0_0
   + 0.5535548694305802 edge_8_3_0_0 + 0.5038939883380836 edge_8_4_0_0
   + 2.070107783396947 edge_8_6_0_0 + 2.557038475233295 edge_8_7_0_0
   + 0.2958405141826833 edge_8_9_0_0 + 1.68316085834529 edge_8_10_0_0
   + 2.416321149897429 edge_8_11_0_0 - 0.460174551876518 edge_8_12_0_0
   + 3.332338340544 edge_8_19_0_0 - 0.4932138260699381 edge_9_8_0_0
   + 0.935266503269797 edge_9_10_0_0 + 0.1178160480197887 edge_9_11_0_0
   - 0.4646143055418612 edge_9_12_0_0 + 0.2682585069205201 edge_9_19_0_0
   - 0.2690716541433 edge_10_2_0_0 - 0.8002756225516408 edge_10_3_0_0
   + 0.5307148546696603 edge_10_4_0_0 + 0.0945082226227645 edge_10_6_0_0
   + 0.4034062094365395 edge_10_7_0_0 + 0.4546955915925402 edge_10_8_0_0
   + 3.196750427224566 edge_10_9_0_0 + 0.7135216646712494 edge_10_11_0_0
   - 0.3066716869815799 edge_10_12_0_0 + 1.262305538118 edge_10_19_0_0
   - 0.77104653882402 edge_11_2_0_0 - 0.3335523435184981 edge_11_3_0_0
   - 0.3098356945417228 edge_11_4_0_0 + 0.1247255536782383 edge_11_6_0_0
   + 0.0734616747495931 edge_11_7_0_0 + 0.0599915453697467 edge_11_8_0_0
   + 1.783202483014026 edge_11_9_0_0 + 1.358600491798695 edge_11_10_0_0
   + 0.3830724569328972 edge_11_12_0_0 + 0.4269417241552611 edge_11_18_0_0
   + 1.6775073024658 edge_11_19_0_0 + 0.2025065481883564 edge_18_12_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_8_2_0_0 + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_9_0_0
   - flow_2_10_0_0 - flow_2_19_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 + flow_10_2_0_0
   + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_6_3_0_0
   + edge_7_3_0_0 + edge_8_3_0_0 + edge_10_3_0_0 + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_19_0_0
   + flow_4_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0
   + flow_10_3_0_0 + flow_11_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0
   + edge_7_4_0_0 + edge_8_4_0_0 + edge_10_4_0_0 + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_6_0_0 - flow_4_7_0_0
   - flow_4_8_0_0 + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_10_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_10_6_0_0 + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 - flow_6_19_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_10_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0 + edge_6_7_0_0
   + edge_8_7_0_0 + edge_10_7_0_0 + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_2_7_0_0 + flow_3_7_0_0 + flow_4_7_0_0
   + flow_6_7_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_6_0_0 - flow_7_9_0_0 - flow_7_11_0_0 - flow_7_19_0_0
   + flow_8_7_0_0 + flow_10_7_0_0 + flow_11_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_6_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_6_8_0_0 - flow_8_2_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0
   - flow_8_10_0_0 - flow_8_11_0_0 - flow_8_12_0_0 - flow_8_19_0_0
   + flow_9_8_0_0 + flow_10_8_0_0 + flow_11_8_0_0 = 1
 one_parent_9: edge_2_9_0_0 + edge_3_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0
   + edge_8_9_0_0 + edge_10_9_0_0 + edge_11_9_0_0 = 1
 consume_one_flow_unit_9: flow_2_9_0_0 + flow_3_9_0_0 + flow_6_9_0_0
   + flow_7_9_0_0 + flow_8_9_0_0 - flow_9_8_0_0 - flow_9_10_0_0
   - flow_9_11_0_0 - flow_9_12_0_0 - flow_9_19_0_0 + flow_10_9_0_0
   + flow_11_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_6_10_0_0 + edge_8_10_0_0 + edge_9_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_6_10_0_0 + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_2_0_0
   - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_6_0_0 - flow_10_7_0_0
   - flow_10_8_0_0 - flow_10_9_0_0 - flow_10_11_0_0 - flow_10_12_0_0
   - flow_10_19_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_3_11_0_0 + edge_6_11_0_0
   + edge_7_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_3_11_0_0 + flow_6_11_0_0
   + flow_7_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0
   - flow_11_2_0_0 - flow_11_3_0_0 - flow_11_4_0_0 - flow_11_6_0_0
   - flow_11_7_0_0 - flow_11_8_0_0 - flow_11_9_0_0 - flow_11_10_0_0
   - flow_11_12_0_0 - flow_11_18_0_0 - flow_11_19_0_0 = 1
 one_parent_12: edge_8_12_0_0 + edge_9_12_0_0 + edge_10_12_0_0
   + edge_11_12_0_0 + edge_18_12_0_0 = 1
 consume_one_flow_unit_12: flow_8_12_0_0 + flow_9_12_0_0 + flow_10_12_0_0
   + flow_11_12_0_0 + flow_18_12_0_0 = 1
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
 one_parent_18: edge_11_18_0_0 = 1
 consume_one_flow_unit_18: flow_11_18_0_0 - flow_18_12_0_0 = 1
 one_parent_19: edge_2_19_0_0 + edge_3_19_0_0 + edge_6_19_0_0
   + edge_7_19_0_0 + edge_8_19_0_0 + edge_9_19_0_0 + edge_10_19_0_0
   + edge_11_19_0_0 = 1
 consume_one_flow_unit_19: flow_2_19_0_0 + flow_3_19_0_0 + flow_6_19_0_0
   + flow_7_19_0_0 + flow_8_19_0_0 + flow_9_19_0_0 + flow_10_19_0_0
   + flow_11_19_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_6_0_0
   + flow_0_8_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 19
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_19_0_0: flow_2_19_0_0 - 1000 edge_2_19_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_19_0_0: flow_3_19_0_0 - 1000 edge_3_19_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_19_0_0: flow_6_19_0_0 - 1000 edge_6_19_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_19_0_0: flow_7_19_0_0 - 1000 edge_7_19_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_8_19_0_0: flow_8_19_0_0 - 1000 edge_8_19_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_19_0_0: flow_9_19_0_0 - 1000 edge_9_19_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_10_19_0_0: flow_10_19_0_0 - 1000 edge_10_19_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_11_18_0_0: flow_11_18_0_0 - 1000 edge_11_18_0_0 <= 0
 inactive_no_flow_11_19_0_0: flow_11_19_0_0 - 1000 edge_11_19_0_0 <= 0
 inactive_no_flow_18_12_0_0: flow_18_12_0_0 - 1000 edge_18_12_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_0_8_0_0
 edge_0_10_0_0 edge_0_11_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0 edge_2_19_0_0
 edge_3_2_0_0 edge_3_4_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_3_19_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_6_2_0_0
 edge_6_3_0_0 edge_6_4_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0
 edge_6_10_0_0 edge_6_11_0_0 edge_6_19_0_0 edge_7_2_0_0 edge_7_3_0_0
 edge_7_4_0_0 edge_7_6_0_0 edge_7_9_0_0 edge_7_11_0_0 edge_7_19_0_0
 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0 edge_8_6_0_0 edge_8_7_0_0
 edge_8_9_0_0 edge_8_10_0_0 edge_8_11_0_0 edge_8_12_0_0 edge_8_19_0_0
 edge_9_8_0_0 edge_9_10_0_0 edge_9_11_0_0 edge_9_12_0_0 edge_9_19_0_0
 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_6_0_0 edge_10_7_0_0
 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_10_12_0_0 edge_10_19_0_0
 edge_11_2_0_0 edge_11_3_0_0 edge_11_4_0_0 edge_11_6_0_0 edge_11_7_0_0
 edge_11_8_0_0 edge_11_9_0_0 edge_11_10_0_0 edge_11_12_0_0 edge_11_18_0_0
 edge_11_19_0_0 edge_18_12_0_0
End
