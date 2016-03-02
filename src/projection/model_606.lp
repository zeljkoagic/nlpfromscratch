Maximize
  - 2.3024197592864 edge_0_1_0_0 - 0.14429097329398 edge_0_3_0_0
   - 0.9263262676406201 edge_0_4_0_0 - 1.954811433627 edge_0_7_0_0
   - 0.9315026761117 edge_0_11_0_0 - 0.0859486 edge_1_3_0_0
   + 0.2024057933930443 edge_2_2_0_0 + 0.0904276354976354 edge_2_3_0_0
   - 0.00728140576331949 edge_2_10_0_0 + 0.8660926592996968 edge_3_1_0_0
   + 0.5054106635913271 edge_3_2_0_0 + 1.365947020820868 edge_3_3_0_0
   + 2.797983013047779 edge_3_4_0_0 + 1.155880509751114 edge_3_5_0_0
   - 0.2462943033246245 edge_3_6_0_0 + 2.134676327066024 edge_3_7_0_0
   + 1.703538939522628 edge_3_8_0_0 + 0.3291485725621167 edge_3_9_0_0
   + 0.2813886206996937 edge_3_10_0_0 + 0.0511686899385366 edge_3_11_0_0
   - 0.7265459301031733 edge_3_12_0_0 + 4.942177944053823 edge_3_13_0_0
   + 0.1594094442521645 edge_4_1_0_0 + 0.7773622176767533 edge_4_2_0_0
   + 0.6693692795302085 edge_4_3_0_0 + 0.2682325104111306 edge_4_4_0_0
   + 1.385554144686648 edge_4_5_0_0 + 2.113606174753693 edge_4_6_0_0
   + 1.120858584866629 edge_4_7_0_0 + 0.4889456102863 edge_4_8_0_0
   + 0.5978283432083156 edge_4_9_0_0 + 0.4618598377711727 edge_4_10_0_0
   + 1.261835387129474 edge_4_11_0_0 + 0.5724037484800516 edge_4_12_0_0
   + 4.054382204553258 edge_4_13_0_0 - 0.1686890585187171 edge_5_4_0_0
   + 0.3547730925949962 edge_5_6_0_0 + 0.0432706055819925 edge_5_10_0_0
   + 0.1075485110573147 edge_6_10_0_0 - 0.1070902331527071 edge_6_11_0_0
   - 0.7374023288561596 edge_7_1_0_0 - 0.3698609671343176 edge_7_2_0_0
   - 0.2115346936722929 edge_7_3_0_0 - 0.2134775178955062 edge_7_4_0_0
   + 0.0481657323969374 edge_7_5_0_0 - 0.5998309965233805 edge_7_6_0_0
   + 1.361410337879727 edge_7_8_0_0 + 0.990123593810062 edge_7_9_0_0
   + 0.1406769281170722 edge_7_10_0_0 - 0.3212707377229432 edge_7_11_0_0
   + 1.927785718809191 edge_7_13_0_0 - 0.2993973216013086 edge_8_5_0_0
   - 0.3643118088632959 edge_8_7_0_0 - 0.2005721570827989 edge_8_9_0_0
   + 0.1341695564913439 edge_9_2_0_0 + 0.104731261496032 edge_9_3_0_0
   + 0.2692722984775336 edge_9_4_0_0 + 0.2763083881641367 edge_9_5_0_0
   + 0.3618255518955794 edge_9_6_0_0 + 0.438186642812467 edge_9_7_0_0
   + 0.7287168568799489 edge_9_8_0_0 + 0.3831068638171962 edge_9_10_0_0
   + 0.0515448848483639 edge_9_11_0_0 - 0.3589203631992632 edge_9_12_0_0
   + 1.74094648208562 edge_9_13_0_0 + 0.043575005615969 edge_10_2_0_0
   - 0.2266738943106495 edge_10_3_0_0 - 0.283711099634431 edge_10_4_0_0
   + 0.1058664715929191 edge_10_8_0_0 + 0.2037766652012329 edge_10_9_0_0
   - 0.0404723676398835 edge_10_12_0_0 + 0.7755352024581955 edge_10_13_0_0
   - 0.1087584054646662 edge_11_1_0_0 - 0.0827797904084242 edge_11_2_0_0
   - 0.3227644242542037 edge_11_3_0_0 + 0.2077411985528971 edge_11_4_0_0
   + 0.4116306962739958 edge_11_5_0_0 + 0.5053184998131228 edge_11_6_0_0
   - 0.0878018328877473 edge_11_7_0_0 + 0.7281006981728391 edge_11_10_0_0
   + 1.048742807228561 edge_11_12_0_0 + 1.764897475921374 edge_11_13_0_0
   - 0.3099061293859908 edge_12_3_0_0 - 0.2727213415007183 edge_12_6_0_0
   + 0.2093462279748838 edge_12_10_0_0 - 0.9972180447716256 edge_13_12_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0 + edge_7_1_0_0
   + edge_11_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_7_1_0_0 + flow_11_1_0_0 = 1
 one_parent_2: edge_2_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_7_2_0_0
   + edge_9_2_0_0 + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: - flow_2_3_0_0 - flow_2_10_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0 + flow_10_2_0_0
   + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_3_3_0_0
   + edge_4_3_0_0 + edge_7_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0
   + edge_11_3_0_0 + edge_12_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_12_0_0 - flow_3_13_0_0
   + flow_4_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0
   + flow_11_3_0_0 + flow_12_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_3_4_0_0 + edge_4_4_0_0 + edge_5_4_0_0
   + edge_7_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0 + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_3_4_0_0 - flow_4_1_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_10_0_0
   - flow_4_11_0_0 - flow_4_12_0_0 - flow_4_13_0_0 + flow_5_4_0_0
   + flow_7_4_0_0 + flow_9_4_0_0 + flow_10_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: edge_3_5_0_0 + edge_4_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0
   + edge_9_5_0_0 + edge_11_5_0_0 = 1
 consume_one_flow_unit_5: flow_3_5_0_0 + flow_4_5_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_10_0_0 + flow_7_5_0_0 + flow_8_5_0_0
   + flow_9_5_0_0 + flow_11_5_0_0 = 1
 one_parent_6: edge_3_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0 + edge_7_6_0_0
   + edge_9_6_0_0 + edge_11_6_0_0 + edge_12_6_0_0 = 1
 consume_one_flow_unit_6: flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0
   - flow_6_10_0_0 - flow_6_11_0_0 + flow_7_6_0_0 + flow_9_6_0_0
   + flow_11_6_0_0 + flow_12_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0 + edge_8_7_0_0
   + edge_9_7_0_0 + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_3_7_0_0 + flow_4_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0
   - flow_7_10_0_0 - flow_7_11_0_0 - flow_7_13_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 + flow_11_7_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_4_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0
   + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_4_8_0_0 + flow_7_8_0_0
   - flow_8_5_0_0 - flow_8_7_0_0 - flow_8_9_0_0 + flow_9_8_0_0
   + flow_10_8_0_0 = 1
 one_parent_9: edge_3_9_0_0 + edge_4_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_3_9_0_0 + flow_4_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0
   - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 - flow_9_11_0_0 - flow_9_12_0_0 - flow_9_13_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_2_10_0_0 + edge_3_10_0_0 + edge_4_10_0_0
   + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_9_10_0_0
   + edge_11_10_0_0 + edge_12_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_3_10_0_0 + flow_4_10_0_0
   + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_9_10_0_0
   - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_8_0_0
   - flow_10_9_0_0 - flow_10_12_0_0 - flow_10_13_0_0 + flow_11_10_0_0
   + flow_12_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_6_11_0_0 + edge_7_11_0_0 + edge_9_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_6_11_0_0 + flow_7_11_0_0 + flow_9_11_0_0 - flow_11_1_0_0
   - flow_11_2_0_0 - flow_11_3_0_0 - flow_11_4_0_0 - flow_11_5_0_0
   - flow_11_6_0_0 - flow_11_7_0_0 - flow_11_10_0_0 - flow_11_12_0_0
   - flow_11_13_0_0 = 1
 one_parent_12: edge_3_12_0_0 + edge_4_12_0_0 + edge_9_12_0_0
   + edge_10_12_0_0 + edge_11_12_0_0 + edge_13_12_0_0 = 1
 consume_one_flow_unit_12: flow_3_12_0_0 + flow_4_12_0_0 + flow_9_12_0_0
   + flow_10_12_0_0 + flow_11_12_0_0 - flow_12_3_0_0 - flow_12_6_0_0
   - flow_12_10_0_0 + flow_13_12_0_0 = 1
 one_parent_13: edge_3_13_0_0 + edge_4_13_0_0 + edge_7_13_0_0
   + edge_9_13_0_0 + edge_10_13_0_0 + edge_11_13_0_0 = 1
 consume_one_flow_unit_13: flow_3_13_0_0 + flow_4_13_0_0 + flow_7_13_0_0
   + flow_9_13_0_0 + flow_10_13_0_0 + flow_11_13_0_0 - flow_13_12_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_7_0_0
   + flow_0_11_0_0 = 13
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_2_2_0_0: flow_2_2_0_0 - 1000 edge_2_2_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_3_0_0: flow_3_3_0_0 - 1000 edge_3_3_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_3_13_0_0: flow_3_13_0_0 - 1000 edge_3_13_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_4_0_0: flow_4_4_0_0 - 1000 edge_4_4_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_4_13_0_0: flow_4_13_0_0 - 1000 edge_4_13_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
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
 inactive_no_flow_7_13_0_0: flow_7_13_0_0 - 1000 edge_7_13_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_13_0_0: flow_9_13_0_0 - 1000 edge_9_13_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_10_13_0_0: flow_10_13_0_0 - 1000 edge_10_13_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_11_13_0_0: flow_11_13_0_0 - 1000 edge_11_13_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_6_0_0: flow_12_6_0_0 - 1000 edge_12_6_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_13_12_0_0: flow_13_12_0_0 - 1000 edge_13_12_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_7_0_0 edge_0_11_0_0
 edge_1_3_0_0 edge_2_2_0_0 edge_2_3_0_0 edge_2_10_0_0 edge_3_1_0_0
 edge_3_2_0_0 edge_3_3_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0
 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_3_11_0_0
 edge_3_12_0_0 edge_3_13_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_4_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0
 edge_4_9_0_0 edge_4_10_0_0 edge_4_11_0_0 edge_4_12_0_0 edge_4_13_0_0
 edge_5_4_0_0 edge_5_6_0_0 edge_5_10_0_0 edge_6_10_0_0 edge_6_11_0_0
 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0
 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0 edge_7_11_0_0
 edge_7_13_0_0 edge_8_5_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_9_2_0_0
 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0 edge_9_6_0_0 edge_9_7_0_0
 edge_9_8_0_0 edge_9_10_0_0 edge_9_11_0_0 edge_9_12_0_0 edge_9_13_0_0
 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_8_0_0 edge_10_9_0_0
 edge_10_12_0_0 edge_10_13_0_0 edge_11_1_0_0 edge_11_2_0_0 edge_11_3_0_0
 edge_11_4_0_0 edge_11_5_0_0 edge_11_6_0_0 edge_11_7_0_0 edge_11_10_0_0
 edge_11_12_0_0 edge_11_13_0_0 edge_12_3_0_0 edge_12_6_0_0 edge_12_10_0_0
 edge_13_12_0_0
End
