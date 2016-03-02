Maximize
  - 3.1473448966743 edge_0_1_0_0 - 2.2477598651344 edge_0_2_0_0
   - 0.28881243171665 edge_0_3_0_0 - 4.945279379504 edge_0_4_0_0
   - 4.042408497402399 edge_0_5_0_0 - 0.517963428435 edge_0_6_0_0
   - 2.712238878434 edge_0_9_0_0 - 2.459116627504 edge_0_10_0_0
   - 5.19284 edge_0_11_0_0 + 0.0450725638327636 edge_1_2_0_0
   - 0.1737586725701811 edge_1_3_0_0 - 0.2729545001026252 edge_1_4_0_0
   - 0.4319876537287511 edge_1_6_0_0 - 1.263430227036276 edge_1_8_0_0
   - 1.171914724811546 edge_1_10_0_0 + 0.0433510673535589 edge_2_1_0_0
   + 2.57082 edge_2_3_0_0 + 0.3365965804272 edge_2_4_0_0
   + 0.81988631661817 edge_2_5_0_0 + 1.0890412670225 edge_2_6_0_0
   - 0.5147885165797198 edge_2_9_0_0 + 0.153003196729 edge_2_10_0_0
   + 1.57165 edge_2_11_0_0 + 1.410780982579872 edge_3_1_0_0
   + 2.69196684975403 edge_3_2_0_0 + 0.4059707771475586 edge_3_4_0_0
   + 1.1612835456003 edge_3_5_0_0 + 2.703216687224578 edge_3_6_0_0
   + 0.1473633734127043 edge_3_7_0_0 - 7.1285141084742978e-04 edge_3_8_0_0
   + 1.138095701225528 edge_3_9_0_0 + 1.766058565283704 edge_3_10_0_0
   + 3.414901731377732 edge_3_11_0_0 - 0.6598328919683201 edge_4_2_0_0
   + 0.47450303377312 edge_4_3_0_0 + 1.170069043869815 edge_4_5_0_0
   + 0.4657302439690977 edge_4_6_0_0 - 0.140356734198487 edge_4_8_0_0
   - 0.1128205177002354 edge_4_9_0_0 + 0.2065512563480571 edge_4_10_0_0
   - 0.044697454021472 edge_4_11_0_0 - 0.4175560908267489 edge_5_1_0_0
   + 0.6450118623029599 edge_5_2_0_0 + 0.93786439212766 edge_5_3_0_0
   + 2.552116081704245 edge_5_4_0_0 + 2.650200031540106 edge_5_6_0_0
   - 0.7224171908949868 edge_5_7_0_0 + 0.03976997296458 edge_5_8_0_0
   + 0.4397014661623327 edge_5_9_0_0 + 1.496420382632846 edge_5_10_0_0
   + 2.23383236561 edge_5_11_0_0 + 0.1129403602427652 edge_6_1_0_0
   + 0.2208881946923321 edge_6_2_0_0 + 0.3326389786415373 edge_6_3_0_0
   + 2.929510394633356 edge_6_4_0_0 + 2.946103196967653 edge_6_5_0_0
   + 0.1844844480410527 edge_6_6_0_0 + 0.6391276913337963 edge_6_7_0_0
   + 0.4492726738243019 edge_6_8_0_0 + 1.346684011701716 edge_6_9_0_0
   + 2.943362997949151 edge_6_10_0_0 + 2.2170250700565 edge_6_11_0_0
   - 0.4566960026444858 edge_7_9_0_0 - 0.4258169396408084 edge_8_4_0_0
   + 0.8489364968814089 edge_8_9_0_0 + 0.133470932474691 edge_8_10_0_0
   + 0.0201771156105786 edge_8_11_0_0 - 0.6795397474375102 edge_9_1_0_0
   - 0.0306881916568168 edge_9_2_0_0 - 0.3306469920625663 edge_9_3_0_0
   + 0.3120732942432137 edge_9_4_0_0 - 0.0216167394379419 edge_9_5_0_0
   + 0.1714774939879494 edge_9_6_0_0 + 0.7173026200818235 edge_9_7_0_0
   + 3.424502920293079 edge_9_8_0_0 + 0.9455875573330226 edge_9_10_0_0
   + 1.71523693798182 edge_9_11_0_0 - 0.6674805170321694 edge_10_1_0_0
   - 0.23891974566475 edge_10_2_0_0 + 0.290306475883 edge_10_3_0_0
   + 0.8073785380747364 edge_10_4_0_0 + 1.228282091133867 edge_10_5_0_0
   + 2.017162784816982 edge_10_6_0_0 + 3.268296211577668 edge_10_7_0_0
   + 0.4192445872887232 edge_10_8_0_0 + 0.4564838754019633 edge_10_9_0_0
   + 2.1070789377775 edge_10_11_0_0 + 0.084206057915875 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_6_0_0 - flow_1_8_0_0 - flow_1_10_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0
   + flow_9_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 - flow_2_11_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_9_2_0_0
   + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   - flow_3_10_0_0 - flow_3_11_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0
   + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0
   - flow_4_6_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_10_0_0
   - flow_4_11_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_8_4_0_0
   + flow_9_4_0_0 + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_9_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0
   - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 - flow_5_10_0_0 - flow_5_11_0_0 + flow_6_5_0_0
   + flow_9_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_6_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_3_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_9_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_3_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_9_0_0 + flow_9_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_1_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_1_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 - flow_8_4_0_0 - flow_8_9_0_0
   - flow_8_10_0_0 - flow_8_11_0_0 + flow_9_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0
   - flow_9_4_0_0 - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_7_0_0
   - flow_9_8_0_0 - flow_9_10_0_0 - flow_9_11_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_2_0_0
   - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_5_0_0 - flow_10_6_0_0
   - flow_10_7_0_0 - flow_10_8_0_0 - flow_10_9_0_0 - flow_10_11_0_0
   + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_2_11_0_0 + edge_3_11_0_0
   + edge_4_11_0_0 + edge_5_11_0_0 + edge_6_11_0_0 + edge_8_11_0_0
   + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_2_11_0_0 + flow_3_11_0_0
   + flow_4_11_0_0 + flow_5_11_0_0 + flow_6_11_0_0 + flow_8_11_0_0
   + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_9_0_0 + flow_0_10_0_0
   + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
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
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
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
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
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
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_6_0_0 edge_1_8_0_0 edge_1_10_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_9_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_4_10_0_0
 edge_4_11_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0
 edge_5_11_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0
 edge_6_5_0_0 edge_6_6_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0
 edge_6_10_0_0 edge_6_11_0_0 edge_7_9_0_0 edge_8_4_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_8_11_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0
 edge_9_4_0_0 edge_9_5_0_0 edge_9_6_0_0 edge_9_7_0_0 edge_9_8_0_0
 edge_9_10_0_0 edge_9_11_0_0 edge_10_1_0_0 edge_10_2_0_0 edge_10_3_0_0
 edge_10_4_0_0 edge_10_5_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_8_0_0
 edge_10_9_0_0 edge_10_11_0_0 edge_11_10_0_0
End
