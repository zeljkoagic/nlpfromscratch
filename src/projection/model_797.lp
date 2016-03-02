Maximize
  - 2.572216441218 edge_0_1_0_0 - 2.08702 edge_0_2_0_0
   - 1.689672926952 edge_0_3_0_0 - 0.9589584136144 edge_0_5_0_0
   - 1.333239107478 edge_0_6_0_0 - 1.3673760892932 edge_0_7_0_0
   - 2.0923184858506 edge_0_8_0_0 - 2.0428687620416 edge_0_10_0_0
   - 4.0192578295996 edge_0_11_0_0 - 0.10808086840882 edge_1_2_0_0
   - 0.3417954216427781 edge_1_3_0_0 - 0.3433153546374029 edge_1_5_0_0
   - 0.5631374420076143 edge_1_6_0_0 - 0.7069035144928463 edge_1_7_0_0
   - 1.308110041696893 edge_1_8_0_0 - 1.537603900805796 edge_1_10_0_0
   - 0.5892663868119793 edge_1_11_0_0 + 0.6091044845747899 edge_2_1_0_0
   + 2.142978714212 edge_2_3_0_0 + 0.3824847653058814 edge_2_4_0_0
   + 1.3953071675207 edge_2_6_0_0 + 0.97716082430456 edge_2_7_0_0
   + 0.73155195610688 edge_2_8_0_0 + 0.1130077064144 edge_2_9_0_0
   + 0.28531481023228 edge_2_10_0_0 + 1.5931391397044 edge_2_11_0_0
   - 0.2033167388498076 edge_3_1_0_0 - 0.0945914432451 edge_3_2_0_0
   + 0.428422497729464 edge_3_4_0_0 + 0.3648278733045056 edge_3_5_0_0
   + 1.046964429487102 edge_3_6_0_0 + 0.7343724733036194 edge_3_7_0_0
   - 0.0426449009068401 edge_3_8_0_0 + 0.3308040257682018 edge_3_10_0_0
   + 0.0574659344887786 edge_3_11_0_0 - 0.5354905643445488 edge_4_2_0_0
   - 0.4206265843423758 edge_4_6_0_0 - 0.2447892881379788 edge_5_1_0_0
   + 0.2717034432123838 edge_5_3_0_0 + 0.1159045456191839 edge_5_6_0_0
   - 0.00132825074864959 edge_5_7_0_0 - 0.0855622268321911 edge_5_8_0_0
   + 0.0987588198698844 edge_5_10_0_0 + 0.0606572520646345 edge_6_1_0_0
   + 1.004100743810968 edge_6_2_0_0 + 3.012264052623431 edge_6_3_0_0
   + 1.313254908096675 edge_6_4_0_0 + 0.816570540214911 edge_6_5_0_0
   + 0.6335109092207678 edge_6_6_0_0 + 1.95872364454613 edge_6_7_0_0
   + 1.048727808154428 edge_6_8_0_0 + 0.4412415451114451 edge_6_9_0_0
   + 1.344557420367293 edge_6_10_0_0 + 1.848245249999526 edge_6_11_0_0
   + 0.2366439672849979 edge_7_1_0_0 - 0.40338445059416 edge_7_2_0_0
   + 0.9286408203510579 edge_7_3_0_0 - 0.1649525901898808 edge_7_4_0_0
   + 0.668556749808627 edge_7_5_0_0 + 1.398616722952732 edge_7_6_0_0
   + 1.244112062514872 edge_7_8_0_0 + 0.2879142373075353 edge_7_9_0_0
   + 2.912465254619449 edge_7_10_0_0 + 0.9490364895682588 edge_7_11_0_0
   - 0.8949587488926218 edge_8_1_0_0 - 0.75980095441194 edge_8_2_0_0
   - 0.4765776854585371 edge_8_3_0_0 - 0.4527358857176276 edge_8_4_0_0
   - 0.2069390399081995 edge_8_5_0_0 - 0.5723928036966097 edge_8_6_0_0
   + 0.5504834878931487 edge_8_7_0_0 + 0.4965085232030869 edge_8_9_0_0
   + 1.119590747570344 edge_8_10_0_0 + 0.9299684420186302 edge_8_11_0_0
   - 0.5483820806508395 edge_9_7_0_0 + 0.2177654584268717 edge_9_8_0_0
   + 0.2712135518806029 edge_9_11_0_0 - 0.6696712731173964 edge_10_1_0_0
   - 0.0690105802410121 edge_10_2_0_0 - 0.0580033233229731 edge_10_3_0_0
   + 0.0855535908266112 edge_10_4_0_0 + 0.0653584191380985 edge_10_5_0_0
   + 0.3642288595044946 edge_10_6_0_0 + 0.74449659798353 edge_10_7_0_0
   + 3.333503354326282 edge_10_8_0_0 + 3.532477298386724 edge_10_9_0_0
   + 1.552932685656097 edge_10_11_0_0 - 1.632869020278053 edge_11_8_0_0
   - 1.440453215436644 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0
   - flow_1_10_0_0 - flow_1_11_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0
   + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_10_0_0 - flow_2_11_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0
   + flow_8_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_10_0_0
   - flow_3_11_0_0 + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0
   + flow_8_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0
   + edge_8_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 - flow_4_2_0_0
   - flow_4_6_0_0 + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   - flow_5_1_0_0 - flow_5_3_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_8_0_0 - flow_5_10_0_0 + flow_6_5_0_0 + flow_7_5_0_0
   + flow_8_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_6_6_0_0 + edge_7_6_0_0
   + edge_8_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 + flow_7_6_0_0 + flow_8_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0
   - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0
   - flow_7_11_0_0 + flow_8_7_0_0 + flow_9_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0
   + edge_10_8_0_0 + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0
   - flow_8_10_0_0 - flow_8_11_0_0 + flow_9_8_0_0 + flow_10_8_0_0
   + flow_11_8_0_0 = 1
 one_parent_9: edge_2_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_2_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_11_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0
   + edge_8_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   + flow_8_10_0_0 - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0
   - flow_10_4_0_0 - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_7_0_0
   - flow_10_8_0_0 - flow_10_9_0_0 - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0 + edge_8_11_0_0
   + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0 + flow_8_11_0_0
   + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_8_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_10_0_0
   + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_6_0_0: flow_6_6_0_0 - 1000 edge_6_6_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_5_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0
 edge_1_10_0_0 edge_1_11_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0
 edge_2_11_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_10_0_0 edge_3_11_0_0
 edge_4_2_0_0 edge_4_6_0_0 edge_5_1_0_0 edge_5_3_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_10_0_0 edge_6_1_0_0 edge_6_2_0_0
 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_6_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_7_1_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0 edge_7_11_0_0 edge_8_1_0_0
 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0
 edge_8_7_0_0 edge_8_9_0_0 edge_8_10_0_0 edge_8_11_0_0 edge_9_7_0_0
 edge_9_8_0_0 edge_9_11_0_0 edge_10_1_0_0 edge_10_2_0_0 edge_10_3_0_0
 edge_10_4_0_0 edge_10_5_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_8_0_0
 edge_10_9_0_0 edge_10_11_0_0 edge_11_8_0_0 edge_11_10_0_0
End
