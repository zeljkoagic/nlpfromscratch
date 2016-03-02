Maximize
  - 2.9264754711885 edge_0_2_0_0 - 2.2334374688088 edge_0_3_0_0
   - 2.100005005509 edge_0_4_0_0 - 3.516473350200001 edge_0_5_0_0
   - 3.035357871987 edge_0_7_0_0 - 3.4307189352564 edge_0_8_0_0
   - 2.4574967978988 edge_0_9_0_0 - 2.229695381549 edge_0_12_0_0
   + 0.0548139178841245 edge_2_3_0_0 - 0.0238013355070193 edge_2_4_0_0
   - 1.399588400874337 edge_2_5_0_0 - 0.6648518159792006 edge_2_7_0_0
   - 1.308222036305377 edge_2_8_0_0 - 0.4945208457398824 edge_2_9_0_0
   - 0.5936688004467348 edge_2_12_0_0 + 0.2833993565113264 edge_2_13_0_0
   + 1.180512443275878 edge_3_2_0_0 - 0.2198085935128131 edge_3_4_0_0
   - 0.750190441037298 edge_3_5_0_0 - 0.5143191813691751 edge_3_7_0_0
   - 0.4815444633489209 edge_3_8_0_0 + 0.1764592644425272 edge_3_9_0_0
   - 0.6182061148345765 edge_3_12_0_0 + 0.1886907696636969 edge_3_13_0_0
   + 0.1675420046755955 edge_4_2_0_0 + 0.1424164930454255 edge_4_3_0_0
   - 0.5087003724205534 edge_4_5_0_0 - 0.5630818689276201 edge_4_7_0_0
   - 0.5014972534564074 edge_4_8_0_0 - 0.3384149437422992 edge_4_9_0_0
   - 0.3282546426334355 edge_4_12_0_0 + 0.7676550580983056 edge_4_13_0_0
   - 0.2604360020530673 edge_5_2_0_0 + 0.7142745975245095 edge_5_3_0_0
   + 0.3253692525480621 edge_5_4_0_0 + 1.893594649828488 edge_5_7_0_0
   + 2.405547512387586 edge_5_8_0_0 + 0.5547453362529569 edge_5_9_0_0
   + 1.639487909792152 edge_5_12_0_0 + 0.6834244360110193 edge_5_13_0_0
   - 0.1163297277623106 edge_7_5_0_0 - 0.1589369997851194 edge_7_8_0_0
   - 0.4521629113333821 edge_7_9_0_0 - 0.2405367729899456 edge_8_2_0_0
   + 0.0865173777765588 edge_8_3_0_0 + 0.251994378157018 edge_8_4_0_0
   - 0.3107351350985559 edge_8_5_0_0 + 0.4274629913502979 edge_8_7_0_0
   + 0.2453568059257617 edge_8_9_0_0 + 1.30574513741463 edge_8_12_0_0
   + 0.3233859384338964 edge_8_13_0_0 - 0.4703935835746778 edge_9_2_0_0
   - 0.6279138692277071 edge_9_3_0_0 - 0.5941310666770891 edge_9_4_0_0
   - 0.4983340868774037 edge_9_5_0_0 - 0.5040642651110785 edge_9_7_0_0
   - 0.2059487886720733 edge_9_8_0_0 + 0.9322243394866022 edge_9_12_0_0
   + 0.5755333457884653 edge_9_13_0_0 + 0.1680608513064691 edge_12_2_0_0
   + 0.3852368670504362 edge_12_3_0_0 + 0.065388228068827 edge_12_4_0_0
   - 0.1692767066592981 edge_12_5_0_0 + 0.0547344514910064 edge_12_7_0_0
   + 0.3360505935880863 edge_12_8_0_0 + 2.846748425970384 edge_12_9_0_0
   + 0.8715656116950846 edge_12_13_0_0 - 0.5375104641482326 edge_13_4_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_8_2_0_0 + edge_9_2_0_0 + edge_12_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_9_0_0
   - flow_2_12_0_0 - flow_2_13_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_8_2_0_0 + flow_9_2_0_0 + flow_12_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_8_3_0_0 + edge_9_3_0_0 + edge_12_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 - flow_3_12_0_0 - flow_3_13_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_8_3_0_0 + flow_9_3_0_0 + flow_12_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_8_4_0_0 + edge_9_4_0_0 + edge_12_4_0_0 + edge_13_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_7_0_0
   - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_12_0_0 - flow_4_13_0_0
   + flow_5_4_0_0 + flow_8_4_0_0 + flow_9_4_0_0 + flow_12_4_0_0
   + flow_13_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 + edge_12_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0 - flow_5_12_0_0
   - flow_5_13_0_0 + flow_7_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0
   + flow_12_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 + edge_12_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 - flow_7_5_0_0 - flow_7_8_0_0
   - flow_7_9_0_0 + flow_8_7_0_0 + flow_9_7_0_0 + flow_12_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_5_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 + edge_12_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_5_8_0_0 + flow_7_8_0_0 - flow_8_2_0_0
   - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_7_0_0
   - flow_8_9_0_0 - flow_8_12_0_0 - flow_8_13_0_0 + flow_9_8_0_0
   + flow_12_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 + edge_12_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0 - flow_9_5_0_0
   - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_12_0_0 - flow_9_13_0_0
   + flow_12_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: edge_0_12_0_0 + edge_2_12_0_0 + edge_3_12_0_0
   + edge_4_12_0_0 + edge_5_12_0_0 + edge_8_12_0_0 + edge_9_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_2_12_0_0 + flow_3_12_0_0
   + flow_4_12_0_0 + flow_5_12_0_0 + flow_8_12_0_0 + flow_9_12_0_0
   - flow_12_2_0_0 - flow_12_3_0_0 - flow_12_4_0_0 - flow_12_5_0_0
   - flow_12_7_0_0 - flow_12_8_0_0 - flow_12_9_0_0 - flow_12_13_0_0 = 1
 one_parent_13: edge_2_13_0_0 + edge_3_13_0_0 + edge_4_13_0_0
   + edge_5_13_0_0 + edge_8_13_0_0 + edge_9_13_0_0 + edge_12_13_0_0 = 1
 consume_one_flow_unit_13: flow_2_13_0_0 + flow_3_13_0_0 + flow_4_13_0_0
   + flow_5_13_0_0 + flow_8_13_0_0 + flow_9_13_0_0 + flow_12_13_0_0
   - flow_13_4_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_12_0_0 = 13
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_2_13_0_0: flow_2_13_0_0 - 1000 edge_2_13_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_3_13_0_0: flow_3_13_0_0 - 1000 edge_3_13_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_4_13_0_0: flow_4_13_0_0 - 1000 edge_4_13_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_5_13_0_0: flow_5_13_0_0 - 1000 edge_5_13_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_8_13_0_0: flow_8_13_0_0 - 1000 edge_8_13_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_13_0_0: flow_9_13_0_0 - 1000 edge_9_13_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_4_0_0: flow_12_4_0_0 - 1000 edge_12_4_0_0 <= 0
 inactive_no_flow_12_5_0_0: flow_12_5_0_0 - 1000 edge_12_5_0_0 <= 0
 inactive_no_flow_12_7_0_0: flow_12_7_0_0 - 1000 edge_12_7_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_13_0_0: flow_12_13_0_0 - 1000 edge_12_13_0_0 <= 0
 inactive_no_flow_13_4_0_0: flow_13_4_0_0 - 1000 edge_13_4_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_0_9_0_0 edge_0_12_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_12_0_0
 edge_2_13_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_9_0_0 edge_3_12_0_0 edge_3_13_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0
 edge_4_12_0_0 edge_4_13_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_12_0_0 edge_5_13_0_0
 edge_7_5_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_5_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_8_12_0_0
 edge_8_13_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0
 edge_9_7_0_0 edge_9_8_0_0 edge_9_12_0_0 edge_9_13_0_0 edge_12_2_0_0
 edge_12_3_0_0 edge_12_4_0_0 edge_12_5_0_0 edge_12_7_0_0 edge_12_8_0_0
 edge_12_9_0_0 edge_12_13_0_0 edge_13_4_0_0
End
