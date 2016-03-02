Maximize
  - 0.5999955897132 edge_0_1_0_0 - 2.3319185149645 edge_0_2_0_0
   - 2.5318907852834 edge_0_3_0_0 - 0.9287514879033999 edge_0_4_0_0
   - 4.0157846205048 edge_0_6_0_0 - 0.7930971507228001 edge_0_7_0_0
   - 3.222146167398 edge_0_8_0_0 + 0.0832872474592437 edge_1_2_0_0
   - 0.1645007796296892 edge_1_3_0_0 + 0.9524582105068899 edge_1_4_0_0
   - 0.3847404405416833 edge_1_5_0_0 + 0.438910584104901 edge_1_6_0_0
   + 0.3683283550051281 edge_1_7_0_0 + 1.641585254174903 edge_1_8_0_0
   + 0.2230771777995438 edge_2_1_0_0 - 0.8001479043057824 edge_2_2_0_0
   + 2.426620468225794 edge_2_3_0_0 + 1.976531332541369 edge_2_4_0_0
   + 0.1505890195280473 edge_2_5_0_0 + 0.7597704061187505 edge_2_6_0_0
   + 0.1914758034991688 edge_2_7_0_0 + 1.981535659546934 edge_2_8_0_0
   + 0.1950598701124991 edge_3_1_0_0 + 0.5186147460810444 edge_3_2_0_0
   + 0.5436572301731193 edge_3_4_0_0 + 1.011915405614454 edge_3_5_0_0
   + 0.9457456197316115 edge_3_6_0_0 + 0.9329257799457834 edge_3_7_0_0
   + 0.7176076120806235 edge_3_8_0_0 - 0.6938598386361228 edge_4_1_0_0
   + 2.658617893722946 edge_4_2_0_0 + 3.294360225431961 edge_4_3_0_0
   + 0.8479096170305095 edge_4_4_0_0 - 0.2567196992546625 edge_4_6_0_0
   + 2.185950645909309 edge_4_7_0_0 + 1.677709462686885 edge_4_8_0_0
   + 0.0922432751546787 edge_5_1_0_0 - 0.2440100038093537 edge_5_2_0_0
   + 0.5814647049771308 edge_5_3_0_0 + 0.9535361610098673 edge_5_6_0_0
   - 0.610792124181341 edge_5_7_0_0 + 0.0787994006845452 edge_5_8_0_0
   + 0.1246099435827883 edge_6_1_0_0 + 0.3149509114372643 edge_6_2_0_0
   + 0.545605983198333 edge_6_3_0_0 - 0.1786975197305508 edge_6_4_0_0
   + 2.504078392982016 edge_6_5_0_0 + 0.5984470301929621 edge_6_7_0_0
   + 0.5201117256825144 edge_6_8_0_0 + 0.1021349019645418 edge_7_1_0_0
   + 0.48645826134951 edge_7_2_0_0 + 0.4991574810971185 edge_7_3_0_0
   + 0.6456166337124279 edge_7_4_0_0 + 0.1134161732433953 edge_7_5_0_0
   + 0.4889540401230733 edge_7_6_0_0 + 0.9527197650945557 edge_7_7_0_0
   + 1.584247502154085 edge_7_8_0_0 - 0.7350365505547994 edge_8_1_0_0
   - 0.510099792385154 edge_8_2_0_0 + 0.341670105612817 edge_8_3_0_0
   - 0.1627902703358335 edge_8_4_0_0 + 0.8206186019934796 edge_8_7_0_0
   + 0.1050350030334736 edge_8_8_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0
   - flow_1_8_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_2_2_0_0 + edge_3_2_0_0
   + edge_4_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0
   + edge_8_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_8_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_4_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 + flow_8_4_0_0 = 1
 one_parent_5: edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_1_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_6_0_0
   - flow_5_7_0_0 - flow_5_8_0_0 + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_7_7_0_0
   + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0
   + edge_8_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_2_0_0: flow_2_2_0_0 - 1000 edge_2_2_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_4_0_0: flow_4_4_0_0 - 1000 edge_4_4_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_7_0_0: flow_7_7_0_0 - 1000 edge_7_7_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_8_0_0: flow_8_8_0_0 - 1000 edge_8_8_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0
 edge_1_5_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_2_1_0_0
 edge_2_2_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_4_0_0 edge_4_6_0_0 edge_4_7_0_0
 edge_4_8_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0
 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_7_1_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_7_0_0 edge_7_8_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_7_0_0 edge_8_8_0_0
End
