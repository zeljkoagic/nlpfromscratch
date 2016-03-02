Maximize
  - 3.758199249847801 edge_0_1_0_0 - 3.1672582896796 edge_0_2_0_0
   - 2.50965452731 edge_0_3_0_0 - 2.9561384525403 edge_0_6_0_0
   - 3.1468516850977 edge_0_7_0_0 - 3.189472909827201 edge_0_10_0_0
   - 3.1322635146016 edge_0_11_0_0 + 1.300099405300482 edge_1_2_0_0
   + 0.2351186363585662 edge_1_3_0_0 + 0.0102377928441734 edge_1_5_0_0
   - 0.2103995477415656 edge_1_6_0_0 + 0.0103995368600021 edge_1_7_0_0
   - 0.0294919529499317 edge_1_9_0_0 - 0.4068425818105449 edge_1_10_0_0
   + 0.0763640627183942 edge_1_11_0_0 + 3.553189982034731 edge_2_1_0_0
   + 2.653928625215669 edge_2_3_0_0 + 0.4615011443826673 edge_2_5_0_0
   + 2.337203789132885 edge_2_6_0_0 + 2.451174019785314 edge_2_7_0_0
   - 0.2358988948419473 edge_2_9_0_0 + 1.662247689432963 edge_2_10_0_0
   + 1.897755953642832 edge_2_11_0_0 + 1.862835043813726 edge_3_1_0_0
   + 1.154658222216532 edge_3_2_0_0 + 0.4003465878332898 edge_3_3_0_0
   - 0.2537033791247879 edge_3_5_0_0 + 0.5770283281396056 edge_3_6_0_0
   + 0.6330051121998864 edge_3_7_0_0 + 0.5308194597001783 edge_3_10_0_0
   - 0.3587961853667753 edge_3_11_0_0 - 1.922934561455371 edge_5_2_0_0
   - 0.0788752942057324 edge_5_3_0_0 + 1.060711051291232 edge_5_6_0_0
   + 0.3978986858934638 edge_5_7_0_0 + 0.4870587675208253 edge_5_9_0_0
   + 0.0701179606516984 edge_5_10_0_0 + 0.0115146893175196 edge_5_11_0_0
   + 0.8829897706492303 edge_6_1_0_0 + 0.4068246884566696 edge_6_2_0_0
   + 1.086106377524203 edge_6_3_0_0 + 3.046896906446044 edge_6_5_0_0
   + 2.489457235278871 edge_6_7_0_0 + 0.6817134862224021 edge_6_9_0_0
   + 2.160647125064671 edge_6_10_0_0 + 1.499114565162672 edge_6_11_0_0
   - 1.545833811982313 edge_7_2_0_0 - 0.5368885891924992 edge_7_3_0_0
   - 0.8873854079730812 edge_7_5_0_0 - 0.7235295074749192 edge_7_6_0_0
   - 0.2880672918170411 edge_7_9_0_0 + 0.0126848062418203 edge_7_10_0_0
   - 0.2727867318491859 edge_7_11_0_0 - 0.9772154463506503 edge_9_3_0_0
   - 0.9542316511401565 edge_9_6_0_0 - 0.1007460537538418 edge_9_7_0_0
   + 0.9369088769849799 edge_9_10_0_0 + 0.1002831105679332 edge_9_11_0_0
   - 0.153036312402443 edge_10_1_0_0 - 0.3508899582558561 edge_10_2_0_0
   + 0.3399298746658163 edge_10_3_0_0 + 1.020919788541594 edge_10_5_0_0
   + 0.8736626636022294 edge_10_6_0_0 + 1.198183328032629 edge_10_7_0_0
   + 3.264615204214434 edge_10_9_0_0 + 1.523513503874733 edge_10_11_0_0
   - 2.339978237979071 edge_11_2_0_0 - 1.15117318118479 edge_11_3_0_0
   - 1.84232421364606 edge_11_6_0_0 - 0.879710261234896 edge_11_7_0_0
   - 0.6758784943977925 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_6_1_0_0
   + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_9_0_0
   - flow_1_10_0_0 - flow_1_11_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_6_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 - flow_2_11_0_0 + flow_3_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_10_2_0_0
   + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_3_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 + edge_9_3_0_0
   + edge_10_3_0_0 + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_10_0_0 - flow_3_11_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0
   + flow_11_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_1_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_9_0_0 - flow_5_10_0_0 - flow_5_11_0_0 + flow_6_5_0_0
   + flow_7_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0
   + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_9_0_0
   - flow_6_10_0_0 - flow_6_11_0_0 + flow_7_6_0_0 + flow_9_6_0_0
   + flow_10_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_9_7_0_0 + edge_10_7_0_0
   + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_9_0_0
   - flow_7_10_0_0 - flow_7_11_0_0 + flow_9_7_0_0 + flow_10_7_0_0
   + flow_11_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_1_9_0_0 + edge_2_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0
   + edge_7_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_1_9_0_0 + flow_2_9_0_0 + flow_5_9_0_0
   + flow_6_9_0_0 + flow_7_9_0_0 - flow_9_3_0_0 - flow_9_6_0_0
   - flow_9_7_0_0 - flow_9_10_0_0 - flow_9_11_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0
   + edge_9_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0
   - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_7_0_0 - flow_10_9_0_0
   - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_5_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0
   + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_5_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0
   + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_2_0_0 - flow_11_3_0_0
   - flow_11_6_0_0 - flow_11_7_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_3_0_0: flow_3_3_0_0 - 1000 edge_3_3_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_10_0_0 edge_0_11_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_5_0_0
 edge_1_6_0_0 edge_1_7_0_0 edge_1_9_0_0 edge_1_10_0_0 edge_1_11_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0
 edge_2_9_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_3_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_10_0_0
 edge_3_11_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_6_0_0 edge_5_7_0_0
 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0 edge_6_1_0_0 edge_6_2_0_0
 edge_6_3_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_9_0_0 edge_6_10_0_0
 edge_6_11_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_9_0_0 edge_7_10_0_0 edge_7_11_0_0 edge_9_3_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_10_0_0 edge_9_11_0_0 edge_10_1_0_0 edge_10_2_0_0
 edge_10_3_0_0 edge_10_5_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_9_0_0
 edge_10_11_0_0 edge_11_2_0_0 edge_11_3_0_0 edge_11_6_0_0 edge_11_7_0_0
 edge_11_10_0_0
End
