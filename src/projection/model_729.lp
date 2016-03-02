Maximize
  - 2.4497171002199 edge_0_1_0_0 - 4.934846738908799 edge_0_2_0_0
   - 1.0176948737917 edge_0_3_0_0 - 3.1530792432608 edge_0_6_0_0
   - 3.6391746939339 edge_0_8_0_0 - 3.5867193543904 edge_0_9_0_0
   - 4.156357007420801 edge_0_10_0_0 - 0.2765541920231025 edge_1_2_0_0
   + 0.1273050556156133 edge_1_3_0_0 - 0.4111743994227824 edge_1_4_0_0
   - 0.6526165333858943 edge_1_6_0_0 - 0.2240128987571198 edge_1_8_0_0
   + 0.07799226368364 edge_1_9_0_0 + 0.7482287369102215 edge_1_10_0_0
   - 0.3783645136756737 edge_2_1_0_0 + 0.6424965895156733 edge_2_3_0_0
   + 0.1587181918655266 edge_2_6_0_0 - 0.1917197071254045 edge_2_8_0_0
   - 1.078512536642908 edge_2_9_0_0 + 0.7477041825272862 edge_2_10_0_0
   + 0.8406310835474365 edge_3_1_0_0 + 2.069289501817362 edge_3_2_0_0
   + 0.3552751711263079 edge_3_4_0_0 + 0.2192112583224826 edge_3_5_0_0
   + 2.256720002493815 edge_3_6_0_0 + 1.928919997397014 edge_3_8_0_0
   + 2.0013390995952 edge_3_9_0_0 + 2.086845294841174 edge_3_10_0_0
   - 0.5254767374324759 edge_4_1_0_0 - 0.0369370739951231 edge_4_3_0_0
   + 1.06484574707436 edge_4_6_0_0 + 0.2758801955735362 edge_4_8_0_0
   + 0.44665399529469 edge_4_9_0_0 + 0.0180604010295019 edge_4_10_0_0
   - 1.234914201162775 edge_5_1_0_0 - 1.393393950686773 edge_5_2_0_0
   + 0.4486819724782891 edge_5_8_0_0 + 0.19505307025888 edge_5_9_0_0
   - 0.7667859306077724 edge_5_10_0_0 - 0.078519451808857 edge_6_1_0_0
   + 0.9408139286761815 edge_6_2_0_0 + 1.129577598294966 edge_6_3_0_0
   + 3.3128179575579 edge_6_4_0_0 + 4.028098530292921 edge_6_5_0_0
   + 0.9242211366996396 edge_6_8_0_0 + 1.1589097218616 edge_6_9_0_0
   + 0.9676404260573018 edge_6_10_0_0 - 1.492415137119454 edge_8_1_0_0
   - 0.7896900035058535 edge_8_6_0_0 + 0.3411907914613 edge_8_9_0_0
   + 0.5425145977480684 edge_8_10_0_0 - 0.0933197298399 edge_9_1_0_0
   - 1.068152608334096 edge_9_2_0_0 - 0.4544420364500187 edge_9_3_0_0
   - 0.35852516169123 edge_9_4_0_0 + 0.80473780686288 edge_9_6_0_0
   + 0.5338296640810101 edge_9_8_0_0 + 0.8325184005863201 edge_9_10_0_0
   - 1.912300288855801 edge_10_1_0_0 - 1.519578575437849 edge_10_6_0_0
   - 1.1331091841608 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0
   + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_6_0_0 - flow_1_8_0_0 - flow_1_9_0_0
   - flow_1_10_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_5_1_0_0 + flow_6_1_0_0 + flow_8_1_0_0 + flow_9_1_0_0
   + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_6_0_0 - flow_2_8_0_0 - flow_2_9_0_0
   - flow_2_10_0_0 + flow_3_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0
   + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_6_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0
   + flow_4_3_0_0 + flow_6_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: edge_1_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0 + edge_9_4_0_0
   = 1
 consume_one_flow_unit_4: flow_1_4_0_0 + flow_3_4_0_0 - flow_4_1_0_0
   - flow_4_3_0_0 - flow_4_6_0_0 - flow_4_8_0_0 - flow_4_9_0_0
   - flow_4_10_0_0 + flow_6_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_3_5_0_0 + edge_6_5_0_0 = 1
 consume_one_flow_unit_5: flow_3_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_8_0_0 - flow_5_9_0_0 - flow_5_10_0_0 + flow_6_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_8_0_0
   - flow_6_9_0_0 - flow_6_10_0_0 + flow_8_6_0_0 + flow_9_6_0_0
   + flow_10_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   - flow_8_1_0_0 - flow_8_6_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0
   + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0
   - flow_9_4_0_0 - flow_9_6_0_0 - flow_9_8_0_0 - flow_9_10_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_6_0_0
   - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_6_0_0
   + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_10_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_6_0_0 edge_0_8_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0
 edge_1_6_0_0 edge_1_8_0_0 edge_1_9_0_0 edge_1_10_0_0 edge_2_1_0_0
 edge_2_3_0_0 edge_2_6_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0
 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_4_1_0_0 edge_4_3_0_0
 edge_4_6_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_4_10_0_0 edge_5_1_0_0
 edge_5_2_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_8_1_0_0 edge_8_6_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0
 edge_9_6_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_10_1_0_0 edge_10_6_0_0
 edge_10_9_0_0
End
