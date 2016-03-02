Maximize
  - 2.4421445853158 edge_0_1_0_0 - 2.6496760801144 edge_0_3_0_0
   - 2.2477595954032 edge_0_5_0_0 - 0.3775903553761 edge_0_6_0_0
   - 3.0987196281536 edge_0_7_0_0 - 3.4617702300858 edge_0_8_0_0
   - 0.6106393051538878 edge_1_3_0_0 - 0.0911507304969656 edge_1_5_0_0
   - 0.1536175790144249 edge_1_6_0_0 - 0.2748484621021606 edge_1_7_0_0
   - 0.2531366582153275 edge_1_8_0_0 + 0.836537119693106 edge_3_1_0_0
   + 1.611521572263065 edge_3_5_0_0 + 0.4130225507405847 edge_3_6_0_0
   + 3.394466566178363 edge_3_7_0_0 + 1.586262729257764 edge_3_8_0_0
   + 1.053155264295818 edge_5_1_0_0 + 1.070841110032965 edge_5_3_0_0
   + 0.5486427967632788 edge_5_6_0_0 + 1.422208976008938 edge_5_7_0_0
   + 0.7480490461934475 edge_5_8_0_0 + 0.8090045602910727 edge_6_1_0_0
   + 3.474561300693943 edge_6_3_0_0 + 1.489357240135648 edge_6_5_0_0
   + 1.429820963926464 edge_6_7_0_0 + 3.473588419514679 edge_6_8_0_0
   + 0.8075361847214048 edge_7_1_0_0 + 0.1712698939113703 edge_7_3_0_0
   + 0.1482188932823344 edge_7_5_0_0 - 0.5088867104293875 edge_7_6_0_0
   - 0.6305931749804816 edge_7_8_0_0 - 1.261745033268393 edge_8_1_0_0
   - 1.103833601207812 edge_8_3_0_0 - 1.910166995891679 edge_8_5_0_0
   - 1.588078066105009 edge_8_7_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0 + edge_6_1_0_0
   + edge_7_1_0_0 + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_5_0_0
   - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_8_0_0 + flow_3_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0
   + edge_7_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   - flow_5_1_0_0 - flow_5_3_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_8_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_3_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_3_6_0_0
   + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_3_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_3_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_3_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_1_0_0
   - flow_8_3_0_0 - flow_8_5_0_0 - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_1_3_0_0 edge_1_5_0_0 edge_1_6_0_0 edge_1_7_0_0
 edge_1_8_0_0 edge_3_1_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_5_1_0_0 edge_5_3_0_0 edge_5_6_0_0 edge_5_7_0_0
 edge_5_8_0_0 edge_6_1_0_0 edge_6_3_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_7_1_0_0 edge_7_3_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_8_1_0_0 edge_8_3_0_0 edge_8_5_0_0 edge_8_7_0_0
End
