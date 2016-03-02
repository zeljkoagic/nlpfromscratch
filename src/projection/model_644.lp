Maximize
  - 2.778734006552 edge_0_1_0_0 - 1.458417564031 edge_0_2_0_0
   - 0.7751524295387701 edge_0_3_0_0 - 3.31008363088 edge_0_5_0_0
   - 1.6760153090584 edge_0_7_0_0 - 1.6165921197535 edge_0_8_0_0
   - 5.47622 edge_0_9_0_0 - 0.1629956963209888 edge_1_2_0_0
   - 0.0737701595554007 edge_1_3_0_0 - 0.2066899151535889 edge_1_5_0_0
   - 0.4094862539384911 edge_1_8_0_0 + 0.6718656722732563 edge_1_9_0_0
   + 0.0700650415614901 edge_2_1_0_0 + 0.9761098875152902 edge_2_3_0_0
   + 0.4432332045013697 edge_2_5_0_0 + 0.2314789908008479 edge_2_7_0_0
   + 0.2293039003481659 edge_2_8_0_0 + 0.46682230456516 edge_2_9_0_0
   + 0.6770362588483757 edge_3_1_0_0 + 3.141283651509084 edge_3_2_0_0
   + 0.6266697422471885 edge_3_4_0_0 + 2.778118330523038 edge_3_5_0_0
   + 0.4003263275681526 edge_3_7_0_0 + 1.638489762213929 edge_3_8_0_0
   + 3.494514020680336 edge_3_9_0_0 - 0.9996641669449516 edge_4_1_0_0
   - 0.428179406211074 edge_4_2_0_0 - 1.038284478983315 edge_4_3_0_0
   + 0.0890406376820019 edge_4_5_0_0 - 0.0408785859206528 edge_4_7_0_0
   - 0.0466711768148618 edge_4_8_0_0 - 0.2836680882076467 edge_4_9_0_0
   - 0.1876065020920733 edge_5_1_0_0 + 1.92694772423031 edge_5_2_0_0
   + 0.8165016312420199 edge_5_3_0_0 + 2.093526910864363 edge_5_4_0_0
   + 1.013488493327505 edge_5_7_0_0 + 1.602081767572874 edge_5_8_0_0
   + 1.8912685437221 edge_5_9_0_0 - 0.5582724901834046 edge_7_1_0_0
   - 0.4305561481936422 edge_7_2_0_0 - 0.7895728618822093 edge_7_3_0_0
   + 0.2088059637109274 edge_7_4_0_0 + 0.00524134404805362 edge_7_5_0_0
   + 0.8674671624972649 edge_7_8_0_0 + 0.263763231552064 edge_7_9_0_0
   - 0.5453286141049861 edge_8_1_0_0 + 0.9536600449818349 edge_8_2_0_0
   + 0.1886785490516754 edge_8_3_0_0 + 0.5930691535083997 edge_8_4_0_0
   + 1.126348389320072 edge_8_5_0_0 + 3.091230975390727 edge_8_7_0_0
   + 1.4025190743368 edge_8_9_0_0 - 1.133432945744896 edge_9_5_0_0
   + 0.080523746854292 edge_9_8_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_8_0_0 - flow_1_9_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 + flow_7_1_0_0
   + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_5_0_0 - flow_2_7_0_0 - flow_2_8_0_0
   - flow_2_9_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_7_2_0_0 + flow_8_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_3_4_0_0 + edge_5_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0
   = 1
 consume_one_flow_unit_4: flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0
   - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 + flow_5_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_7_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 + flow_7_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_8_0_0
   - flow_7_9_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_5_0_0 - flow_8_7_0_0 - flow_8_9_0_0 + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_5_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_5_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_5_0_0 - flow_9_8_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0 = 9
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_0_9_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_5_0_0
 edge_1_8_0_0 edge_1_9_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_5_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_7_0_0
 edge_4_8_0_0 edge_4_9_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_7_1_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_8_0_0
 edge_7_9_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_9_5_0_0 edge_9_8_0_0
End
