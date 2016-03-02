Maximize
  - 1.6975154915208 edge_0_1_0_0 - 3.4811970723387 edge_0_2_0_0
   - 1.86523882514 edge_0_3_0_0 - 3.7087021878816 edge_0_4_0_0
   - 3.493532772533 edge_0_5_0_0 - 2.1782486593377 edge_0_7_0_0
   - 2.937549471241 edge_0_10_0_0 - 2.2721972256656 edge_0_11_0_0
   + 2.433539836190768 edge_1_2_0_0 + 1.998860509528268 edge_1_3_0_0
   - 0.2714108693075545 edge_1_4_0_0 - 1.13675658912687 edge_1_5_0_0
   + 0.0199185478301314 edge_1_7_0_0 + 1.325473553623288 edge_1_10_0_0
   + 0.1835424810081862 edge_1_11_0_0 - 0.3915852037919275 edge_2_1_0_0
   + 1.455432407359784 edge_2_3_0_0 + 0.0898790216788909 edge_2_4_0_0
   + 0.463344019061091 edge_2_5_0_0 + 0.3070102062572089 edge_2_7_0_0
   - 0.677818319080603 edge_2_9_0_0 + 1.094251650797667 edge_2_10_0_0
   + 0.85193065306935 edge_2_11_0_0 + 0.8693602123535582 edge_3_1_0_0
   + 2.343712844557225 edge_3_2_0_0 + 0.8882273540902224 edge_3_4_0_0
   + 3.102176473067853 edge_3_5_0_0 + 0.9339309792734815 edge_3_7_0_0
   + 0.8543865538998888 edge_3_9_0_0 + 3.635972628600809 edge_3_10_0_0
   + 2.891367664216 edge_3_11_0_0 - 0.4079865939873467 edge_4_1_0_0
   - 0.8771297376786791 edge_4_2_0_0 - 0.3412771779503251 edge_4_3_0_0
   + 1.443635958895565 edge_4_5_0_0 + 0.4622164899871914 edge_4_7_0_0
   - 0.0403973706080563 edge_4_10_0_0 + 0.0811681094570937 edge_4_11_0_0
   + 0.0163778729237263 edge_5_1_0_0 - 0.118514224975646 edge_5_2_0_0
   + 0.5240240533050652 edge_5_3_0_0 + 3.54255921892053 edge_5_4_0_0
   + 0.7395499106387533 edge_5_9_0_0 + 0.5769306910030292 edge_5_10_0_0
   + 0.5470627326197 edge_5_11_0_0 + 0.2144747517085485 edge_7_1_0_0
   - 0.3108672338193379 edge_7_2_0_0 - 0.0478953202920722 edge_7_3_0_0
   + 1.010997210144952 edge_7_4_0_0 + 0.7982122373409088 edge_7_10_0_0
   - 1.166155408002511 edge_9_1_0_0 + 0.1337250166744827 edge_9_2_0_0
   - 0.3583079757216038 edge_9_3_0_0 + 0.3654314184712898 edge_9_5_0_0
   - 0.6037576026015601 edge_9_11_0_0 - 0.8659924936585309 edge_10_1_0_0
   + 1.336969798844393 edge_10_2_0_0 + 1.504740041278744 edge_10_3_0_0
   + 3.052403944071419 edge_10_4_0_0 + 2.549882726821527 edge_10_5_0_0
   + 3.000817899359212 edge_10_7_0_0 + 3.58461772478052 edge_10_9_0_0
   + 1.909971646120476 edge_10_11_0_0 - 0.3934368860861222 edge_11_1_0_0
   + 0.0331553119794932 edge_11_3_0_0 + 0.00770341099698525 edge_11_4_0_0
   - 0.254271354255576 edge_11_5_0_0 + 0.479014281498606 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_7_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0
   + edge_11_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_7_0_0 - flow_1_10_0_0
   - flow_1_11_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_7_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0
   + flow_11_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_7_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_7_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 - flow_2_11_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0
   + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_7_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0
   + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_7_0_0 - flow_3_9_0_0 - flow_3_10_0_0 - flow_3_11_0_0
   + flow_4_3_0_0 + flow_5_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0
   + flow_10_3_0_0 + flow_11_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_7_4_0_0 + edge_10_4_0_0 + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_7_0_0 - flow_4_10_0_0 - flow_4_11_0_0
   + flow_5_4_0_0 + flow_7_4_0_0 + flow_10_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_9_5_0_0 + edge_10_5_0_0 + edge_11_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_9_0_0 - flow_5_10_0_0
   - flow_5_11_0_0 + flow_9_5_0_0 + flow_10_5_0_0 + flow_11_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_10_0_0 + flow_10_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_2_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0 + edge_10_9_0_0
   = 1
 consume_one_flow_unit_9: flow_2_9_0_0 + flow_3_9_0_0 + flow_5_9_0_0
   - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_5_0_0
   - flow_9_11_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_7_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_7_10_0_0
   - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_5_0_0 - flow_10_7_0_0 - flow_10_9_0_0 - flow_10_11_0_0
   + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_4_11_0_0 + edge_5_11_0_0 + edge_9_11_0_0
   + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_4_11_0_0 + flow_5_11_0_0 + flow_9_11_0_0
   + flow_10_11_0_0 - flow_11_1_0_0 - flow_11_3_0_0 - flow_11_4_0_0
   - flow_11_5_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_7_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_7_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_1_7_0_0 edge_1_10_0_0 edge_1_11_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_7_0_0
 edge_2_9_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_7_0_0 edge_3_9_0_0 edge_3_10_0_0
 edge_3_11_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0
 edge_4_7_0_0 edge_4_10_0_0 edge_4_11_0_0 edge_5_1_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_4_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0
 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_10_0_0
 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_5_0_0 edge_9_11_0_0
 edge_10_1_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_5_0_0
 edge_10_7_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_11_1_0_0 edge_11_3_0_0
 edge_11_4_0_0 edge_11_5_0_0 edge_11_10_0_0
End
