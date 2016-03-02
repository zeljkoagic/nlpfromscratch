Maximize
  - 2.774078434323 edge_0_1_0_0 - 2.5065196992176 edge_0_2_0_0
   - 0.1889962441888 edge_0_3_0_0 - 3.033648361829 edge_0_4_0_0
   - 3.824729176221 edge_0_5_0_0 - 1.3263927464667 edge_0_6_0_0
   - 2.6182297329144 edge_0_9_0_0 - 1.567102723736 edge_0_11_0_0
   - 4.1832332649836 edge_0_12_0_0 + 1.742520546025949 edge_1_2_0_0
   + 0.5568316469287956 edge_1_3_0_0 + 0.1723220893995062 edge_1_4_0_0
   + 0.1902694964016222 edge_1_6_0_0 - 0.1701536271816227 edge_1_7_0_0
   + 0.0819181691394496 edge_1_9_0_0 - 0.4626562919599234 edge_1_11_0_0
   + 1.094486996833931 edge_1_12_0_0 + 0.8451602136665386 edge_2_1_0_0
   + 1.1607712595445 edge_2_3_0_0 + 0.8727114240101366 edge_2_4_0_0
   + 0.1194038325426 edge_2_5_0_0 + 1.120257658656865 edge_2_6_0_0
   + 0.2580743467404131 edge_2_7_0_0 + 0.395168736083836 edge_2_9_0_0
   + 0.00513823738465397 edge_2_11_0_0 + 1.15277 edge_2_12_0_0
   + 1.884808230814722 edge_3_1_0_0 + 2.3472207772146 edge_3_2_0_0
   + 1.441076452438295 edge_3_4_0_0 + 0.9026354568246383 edge_3_5_0_0
   + 2.703071965641737 edge_3_6_0_0 + 0.448966702043386 edge_3_7_0_0
   + 2.117943936897065 edge_3_9_0_0 + 1.986593771464201 edge_3_11_0_0
   + 3.90060626308548 edge_3_12_0_0 - 0.3681645417689487 edge_4_1_0_0
   - 0.3272687840024812 edge_4_2_0_0 + 0.4592699090975668 edge_4_3_0_0
   + 0.3589141196886 edge_4_5_0_0 + 0.5794835454932664 edge_4_6_0_0
   - 0.0200728607416877 edge_4_7_0_0 + 0.0672170464888493 edge_4_9_0_0
   + 0.1394718217967238 edge_4_11_0_0 + 0.227079066702114 edge_4_12_0_0
   - 0.3382316808165 edge_5_2_0_0 - 0.1686687531920887 edge_5_3_0_0
   + 0.9835451317398 edge_5_4_0_0 + 0.3568910228616702 edge_5_6_0_0
   - 0.7918999744430999 edge_5_12_0_0 + 1.633036364975438 edge_6_1_0_0
   + 1.738908884230167 edge_6_2_0_0 + 0.6023542959926992 edge_6_3_0_0
   + 3.4008520178809 edge_6_4_0_0 + 1.997796616982231 edge_6_5_0_0
   + 1.095122829177376 edge_6_7_0_0 + 2.52284811084983 edge_6_9_0_0
   + 0.965310165899317 edge_6_11_0_0 + 1.83283539957355 edge_6_12_0_0
   - 0.6912874328673696 edge_7_1_0_0 - 0.4040635071950933 edge_7_2_0_0
   - 0.0113538269453782 edge_7_4_0_0 + 0.2033116752256108 edge_7_6_0_0
   + 0.8777884095142543 edge_7_9_0_0 - 0.1281225798784624 edge_7_11_0_0
   + 0.1594780351671345 edge_7_12_0_0 - 0.2894786593542192 edge_9_1_0_0
   - 0.4120757842087474 edge_9_2_0_0 - 0.1971418023648196 edge_9_3_0_0
   + 1.337815097002603 edge_9_4_0_0 + 0.5457500554506476 edge_9_6_0_0
   + 2.870729546942807 edge_9_7_0_0 + 1.198912360090722 edge_9_11_0_0
   + 0.7005091472084661 edge_9_12_0_0 + 0.165488149264759 edge_11_1_0_0
   + 0.2391265286066196 edge_11_2_0_0 + 0.2444766920608682 edge_11_3_0_0
   + 0.8706132395554107 edge_11_4_0_0 + 0.3395989511731025 edge_11_6_0_0
   + 0.8263880383957701 edge_11_7_0_0 + 0.6171816111446551 edge_11_9_0_0
   + 0.6929312320854327 edge_11_12_0_0 - 1.283002537316389 edge_12_1_0_0
   - 2.32243846718975 edge_12_2_0_0 - 0.5959662174821999 edge_12_6_0_0
   - 1.421744147170546 edge_12_7_0_0 - 1.418842977839469 edge_12_9_0_0
   - 0.6126681913996137 edge_12_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_9_1_0_0 + edge_11_1_0_0
   + edge_12_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_9_0_0
   - flow_1_11_0_0 - flow_1_12_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_9_1_0_0
   + flow_11_1_0_0 + flow_12_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 + edge_9_2_0_0
   + edge_11_2_0_0 + edge_12_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_9_0_0 - flow_2_11_0_0 - flow_2_12_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0
   + flow_7_2_0_0 + flow_9_2_0_0 + flow_11_2_0_0 + flow_12_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_9_3_0_0 + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_9_0_0 - flow_3_11_0_0
   - flow_3_12_0_0 + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_9_3_0_0 + flow_11_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 + edge_9_4_0_0
   + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_9_0_0
   - flow_4_11_0_0 - flow_4_12_0_0 + flow_5_4_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 + flow_9_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_12_0_0 + flow_6_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_7_6_0_0 + edge_9_6_0_0
   + edge_11_6_0_0 + edge_12_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_9_0_0 - flow_6_11_0_0 - flow_6_12_0_0
   + flow_7_6_0_0 + flow_9_6_0_0 + flow_11_6_0_0 + flow_12_6_0_0 = 1
 one_parent_7: edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_6_7_0_0 + edge_9_7_0_0 + edge_11_7_0_0 + edge_12_7_0_0 = 1
 consume_one_flow_unit_7: flow_1_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0
   - flow_7_4_0_0 - flow_7_6_0_0 - flow_7_9_0_0 - flow_7_11_0_0
   - flow_7_12_0_0 + flow_9_7_0_0 + flow_11_7_0_0 + flow_12_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_11_9_0_0
   + edge_12_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0
   - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_11_0_0 - flow_9_12_0_0
   + flow_11_9_0_0 + flow_12_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_4_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0
   + edge_9_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_4_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0
   + flow_9_11_0_0 - flow_11_1_0_0 - flow_11_2_0_0 - flow_11_3_0_0
   - flow_11_4_0_0 - flow_11_6_0_0 - flow_11_7_0_0 - flow_11_9_0_0
   - flow_11_12_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_2_12_0_0
   + edge_3_12_0_0 + edge_4_12_0_0 + edge_5_12_0_0 + edge_6_12_0_0
   + edge_7_12_0_0 + edge_9_12_0_0 + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_2_12_0_0
   + flow_3_12_0_0 + flow_4_12_0_0 + flow_5_12_0_0 + flow_6_12_0_0
   + flow_7_12_0_0 + flow_9_12_0_0 + flow_11_12_0_0 - flow_12_1_0_0
   - flow_12_2_0_0 - flow_12_6_0_0 - flow_12_7_0_0 - flow_12_9_0_0
   - flow_12_11_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_9_0_0 + flow_0_11_0_0
   + flow_0_12_0_0 = 12
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_6_0_0: flow_12_6_0_0 - 1000 edge_12_6_0_0 <= 0
 inactive_no_flow_12_7_0_0: flow_12_7_0_0 - 1000 edge_12_7_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_9_0_0 edge_0_11_0_0 edge_0_12_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_9_0_0
 edge_1_11_0_0 edge_1_12_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_9_0_0 edge_2_11_0_0
 edge_2_12_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_9_0_0 edge_3_11_0_0 edge_3_12_0_0
 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0
 edge_4_7_0_0 edge_4_9_0_0 edge_4_11_0_0 edge_4_12_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0 edge_5_12_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_9_0_0 edge_6_11_0_0 edge_6_12_0_0 edge_7_1_0_0 edge_7_2_0_0
 edge_7_4_0_0 edge_7_6_0_0 edge_7_9_0_0 edge_7_11_0_0 edge_7_12_0_0
 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_11_0_0 edge_9_12_0_0 edge_11_1_0_0 edge_11_2_0_0
 edge_11_3_0_0 edge_11_4_0_0 edge_11_6_0_0 edge_11_7_0_0 edge_11_9_0_0
 edge_11_12_0_0 edge_12_1_0_0 edge_12_2_0_0 edge_12_6_0_0 edge_12_7_0_0
 edge_12_9_0_0 edge_12_11_0_0
End
