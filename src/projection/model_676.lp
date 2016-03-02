Maximize
  - 4.9292033460537 edge_0_2_0_0 - 2.5103853008755 edge_0_3_0_0
   - 5.6608132675773 edge_0_5_0_0 - 4.668542043010801 edge_0_6_0_0
   - 4.8279800953248 edge_0_8_0_0 + 0.4411248957539889 edge_2_3_0_0
   - 0.5484865143477169 edge_2_5_0_0 - 0.2571906165666235 edge_2_6_0_0
   - 0.385087178281336 edge_2_8_0_0 + 0.093017078189434 edge_2_9_0_0
   + 2.407329703435377 edge_3_2_0_0 + 0.7606326561709804 edge_3_5_0_0
   + 1.912364533081899 edge_3_6_0_0 + 0.9398326806960576 edge_3_7_0_0
   + 2.070744423056012 edge_3_8_0_0 + 2.5362776803785 edge_3_9_0_0
   - 0.4079635070146609 edge_5_2_0_0 - 0.3960985427683644 edge_5_3_0_0
   + 0.3520647351166442 edge_5_6_0_0 + 0.9072802370951748 edge_5_7_0_0
   + 0.5142477302196209 edge_5_8_0_0 - 0.099310637172075 edge_5_9_0_0
   + 0.0897277370318423 edge_6_2_0_0 + 0.3325146339392198 edge_6_3_0_0
   + 1.713864015890701 edge_6_5_0_0 + 1.05841486375008 edge_6_7_0_0
   + 0.0372964859226228 edge_6_8_0_0 + 0.29064052422936 edge_6_9_0_0
   - 0.1184109257146051 edge_7_5_0_0 - 0.0956754826144223 edge_7_6_0_0
   + 0.807343154139408 edge_7_8_0_0 - 0.25546466814901 edge_7_9_0_0
   + 0.6416146041523175 edge_8_2_0_0 + 0.0481688736395255 edge_8_3_0_0
   + 0.5138032929463733 edge_8_5_0_0 + 0.8809119789652347 edge_8_6_0_0
   + 2.879775300829324 edge_8_7_0_0 + 3.818512956722501 edge_8_8_0_0
   + 0.8179489258936801 edge_8_9_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_8_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_5_0_0
   - flow_2_6_0_0 - flow_2_8_0_0 - flow_2_9_0_0 + flow_3_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_8_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0
   + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 + flow_5_3_0_0 + flow_6_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_8_0_0 - flow_5_9_0_0 + flow_6_5_0_0 + flow_7_5_0_0
   + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 = 1
 one_parent_7: edge_3_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0
   = 1
 consume_one_flow_unit_7: flow_3_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0
   + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_8_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_2_0_0
   - flow_8_3_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0
   - flow_8_9_0_0 = 1
 one_parent_9: edge_2_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0
   + edge_7_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_2_9_0_0 + flow_3_9_0_0 + flow_5_9_0_0
   + flow_6_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_8_0_0 = 9
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_8_0_0: flow_8_8_0_0 - 1000 edge_8_8_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_8_0_0
 edge_2_3_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_8_0_0 edge_2_9_0_0
 edge_3_2_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_6_0_0 edge_5_7_0_0
 edge_5_8_0_0 edge_5_9_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0 edge_7_5_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_7_9_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_5_0_0
 edge_8_6_0_0 edge_8_7_0_0 edge_8_8_0_0 edge_8_9_0_0
End
