Maximize
  - 4.8848365211996 edge_0_1_0_0 - 1.732692156132 edge_0_3_0_0
   - 3.2614153219618 edge_0_4_0_0 - 2.406157317861 edge_0_7_0_0
   - 0.51112240241595 edge_0_8_0_0 - 2.2069626956595 edge_0_9_0_0
   - 2.7433379253462 edge_0_11_0_0 - 1.240693436061 edge_0_14_0_0
   - 3.4481995917702 edge_0_15_0_0 - 0.3042875059045464 edge_1_3_0_0
   - 0.0676844250060901 edge_1_4_0_0 - 0.2515549782504495 edge_1_5_0_0
   + 0.8114662763953181 edge_3_1_0_0 + 1.52839364724558 edge_3_4_0_0
   + 0.1680107466483583 edge_3_5_0_0 + 0.3088019271485232 edge_3_7_0_0
   - 0.3247971439905322 edge_3_8_0_0 - 0.5320053709294096 edge_3_9_0_0
   - 0.2101530842098949 edge_3_11_0_0 + 0.0509014083285034 edge_3_14_0_0
   - 0.6490724290692044 edge_3_15_0_0 - 0.651731608701847 edge_4_1_0_0
   - 0.3014439825006037 edge_4_3_0_0 - 0.0598363756303831 edge_4_5_0_0
   - 0.7378297804705605 edge_4_8_0_0 - 1.024418379313301 edge_4_11_0_0
   - 0.6579310824823076 edge_5_1_0_0 - 0.3963353354944338 edge_5_3_0_0
   - 0.2280797022442274 edge_5_4_0_0 - 1.286279197704158 edge_7_1_0_0
   - 0.2770125981140285 edge_7_3_0_0 - 0.2682745908580461 edge_7_4_0_0
   + 0.2925804468952035 edge_7_5_0_0 + 0.4661650165685052 edge_7_8_0_0
   - 0.0204321831487826 edge_7_9_0_0 - 1.372680101850553 edge_7_10_0_0
   - 0.0509756554990517 edge_7_11_0_0 - 0.273439632799123 edge_7_14_0_0
   + 0.1987031610928779 edge_7_15_0_0 - 1.062326902080628 edge_8_1_0_0
   + 0.7501460393335742 edge_8_3_0_0 - 0.4301783434977987 edge_8_4_0_0
   + 0.1862345002582369 edge_8_5_0_0 + 2.015745991254906 edge_8_7_0_0
   + 0.3356856272830207 edge_8_9_0_0 - 0.0469693410688942 edge_8_10_0_0
   + 1.06011663181303 edge_8_11_0_0 + 0.3115758077596252 edge_8_14_0_0
   + 2.564495917393344 edge_8_15_0_0 - 0.5103048657765062 edge_9_3_0_0
   - 0.2377029565603439 edge_9_7_0_0 - 0.2626048343391057 edge_9_8_0_0
   - 0.0795725538957058 edge_9_11_0_0 - 0.2577815208904399 edge_9_15_0_0
   - 0.6667613456301742 edge_10_11_0_0 + 0.2386322225136413 edge_10_14_0_0
   - 0.1257953717900324 edge_11_1_0_0 + 0.0147514060643843 edge_11_3_0_0
   - 0.5691029267109399 edge_11_4_0_0 - 0.7339532187356259 edge_11_5_0_0
   + 0.7910215119405307 edge_11_7_0_0 - 0.1368845151809081 edge_11_8_0_0
   + 0.845552028448227 edge_11_9_0_0 + 0.4860074410284617 edge_11_10_0_0
   + 0.0812782432919074 edge_11_14_0_0 + 1.060140506508428 edge_11_15_0_0
   - 0.6199758648265866 edge_14_1_0_0 - 0.3208951329837916 edge_14_3_0_0
   - 0.430863880037671 edge_14_4_0_0 - 0.00304934500463133 edge_14_5_0_0
   - 0.1736803143075674 edge_14_7_0_0 + 0.0785478237018347 edge_14_8_0_0
   + 0.5595642660839637 edge_14_10_0_0 + 0.3280947500566194 edge_14_11_0_0
   - 0.5615126408918302 edge_15_9_0_0 - 0.7842192347486374 edge_15_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0 + edge_5_1_0_0
   + edge_7_1_0_0 + edge_8_1_0_0 + edge_11_1_0_0 + edge_14_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_4_0_0
   - flow_1_5_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0
   + flow_7_1_0_0 + flow_8_1_0_0 + flow_11_1_0_0 + flow_14_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_7_3_0_0 + edge_8_3_0_0 + edge_9_3_0_0 + edge_11_3_0_0
   + edge_14_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 - flow_3_11_0_0 - flow_3_14_0_0 - flow_3_15_0_0
   + flow_4_3_0_0 + flow_5_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0
   + flow_9_3_0_0 + flow_11_3_0_0 + flow_14_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_7_4_0_0 + edge_8_4_0_0 + edge_11_4_0_0 + edge_14_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_3_4_0_0
   - flow_4_1_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_8_0_0
   - flow_4_11_0_0 + flow_5_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_11_4_0_0 + flow_14_4_0_0 = 1
 one_parent_5: edge_1_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0 + edge_7_5_0_0
   + edge_8_5_0_0 + edge_11_5_0_0 + edge_14_5_0_0 = 1
 consume_one_flow_unit_5: flow_1_5_0_0 + flow_3_5_0_0 + flow_4_5_0_0
   - flow_5_1_0_0 - flow_5_3_0_0 - flow_5_4_0_0 + flow_7_5_0_0
   + flow_8_5_0_0 + flow_11_5_0_0 + flow_14_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_3_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0
   + edge_11_7_0_0 + edge_14_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_3_7_0_0 - flow_7_1_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_8_0_0
   - flow_7_9_0_0 - flow_7_10_0_0 - flow_7_11_0_0 - flow_7_14_0_0
   - flow_7_15_0_0 + flow_8_7_0_0 + flow_9_7_0_0 + flow_11_7_0_0
   + flow_14_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_7_8_0_0
   + edge_9_8_0_0 + edge_11_8_0_0 + edge_14_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_7_8_0_0 - flow_8_1_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_5_0_0 - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   - flow_8_11_0_0 - flow_8_14_0_0 - flow_8_15_0_0 + flow_9_8_0_0
   + flow_11_8_0_0 + flow_14_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_3_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0
   + edge_11_9_0_0 + edge_15_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_3_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_3_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_11_0_0 - flow_9_15_0_0 + flow_11_9_0_0 + flow_15_9_0_0 = 1
 one_parent_10: edge_7_10_0_0 + edge_8_10_0_0 + edge_11_10_0_0
   + edge_14_10_0_0 = 1
 consume_one_flow_unit_10: flow_7_10_0_0 + flow_8_10_0_0 - flow_10_11_0_0
   - flow_10_14_0_0 + flow_11_10_0_0 + flow_14_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_7_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0
   + edge_14_11_0_0 + edge_15_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_7_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0
   - flow_11_1_0_0 - flow_11_3_0_0 - flow_11_4_0_0 - flow_11_5_0_0
   - flow_11_7_0_0 - flow_11_8_0_0 - flow_11_9_0_0 - flow_11_10_0_0
   - flow_11_14_0_0 - flow_11_15_0_0 + flow_14_11_0_0 + flow_15_11_0_0 = 1
 one_parent_12: = 1
 consume_one_flow_unit_12: = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: edge_0_14_0_0 + edge_3_14_0_0 + edge_7_14_0_0
   + edge_8_14_0_0 + edge_10_14_0_0 + edge_11_14_0_0 = 1
 consume_one_flow_unit_14: flow_0_14_0_0 + flow_3_14_0_0 + flow_7_14_0_0
   + flow_8_14_0_0 + flow_10_14_0_0 + flow_11_14_0_0 - flow_14_1_0_0
   - flow_14_3_0_0 - flow_14_4_0_0 - flow_14_5_0_0 - flow_14_7_0_0
   - flow_14_8_0_0 - flow_14_10_0_0 - flow_14_11_0_0 = 1
 one_parent_15: edge_0_15_0_0 + edge_3_15_0_0 + edge_7_15_0_0
   + edge_8_15_0_0 + edge_9_15_0_0 + edge_11_15_0_0 = 1
 consume_one_flow_unit_15: flow_0_15_0_0 + flow_3_15_0_0 + flow_7_15_0_0
   + flow_8_15_0_0 + flow_9_15_0_0 + flow_11_15_0_0 - flow_15_9_0_0
   - flow_15_11_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_7_0_0
   + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_11_0_0 + flow_0_14_0_0
   + flow_0_15_0_0 = 15
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_14_0_0: flow_0_14_0_0 - 1000 edge_0_14_0_0 <= 0
 inactive_no_flow_0_15_0_0: flow_0_15_0_0 - 1000 edge_0_15_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_14_0_0: flow_3_14_0_0 - 1000 edge_3_14_0_0 <= 0
 inactive_no_flow_3_15_0_0: flow_3_15_0_0 - 1000 edge_3_15_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_14_0_0: flow_7_14_0_0 - 1000 edge_7_14_0_0 <= 0
 inactive_no_flow_7_15_0_0: flow_7_15_0_0 - 1000 edge_7_15_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_14_0_0: flow_8_14_0_0 - 1000 edge_8_14_0_0 <= 0
 inactive_no_flow_8_15_0_0: flow_8_15_0_0 - 1000 edge_8_15_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_15_0_0: flow_9_15_0_0 - 1000 edge_9_15_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_14_0_0: flow_10_14_0_0 - 1000 edge_10_14_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_14_0_0: flow_11_14_0_0 - 1000 edge_11_14_0_0 <= 0
 inactive_no_flow_11_15_0_0: flow_11_15_0_0 - 1000 edge_11_15_0_0 <= 0
 inactive_no_flow_14_1_0_0: flow_14_1_0_0 - 1000 edge_14_1_0_0 <= 0
 inactive_no_flow_14_3_0_0: flow_14_3_0_0 - 1000 edge_14_3_0_0 <= 0
 inactive_no_flow_14_4_0_0: flow_14_4_0_0 - 1000 edge_14_4_0_0 <= 0
 inactive_no_flow_14_5_0_0: flow_14_5_0_0 - 1000 edge_14_5_0_0 <= 0
 inactive_no_flow_14_7_0_0: flow_14_7_0_0 - 1000 edge_14_7_0_0 <= 0
 inactive_no_flow_14_8_0_0: flow_14_8_0_0 - 1000 edge_14_8_0_0 <= 0
 inactive_no_flow_14_10_0_0: flow_14_10_0_0 - 1000 edge_14_10_0_0 <= 0
 inactive_no_flow_14_11_0_0: flow_14_11_0_0 - 1000 edge_14_11_0_0 <= 0
 inactive_no_flow_15_9_0_0: flow_15_9_0_0 - 1000 edge_15_9_0_0 <= 0
 inactive_no_flow_15_11_0_0: flow_15_11_0_0 - 1000 edge_15_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_7_0_0 edge_0_8_0_0
 edge_0_9_0_0 edge_0_11_0_0 edge_0_14_0_0 edge_0_15_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_3_1_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_11_0_0 edge_3_14_0_0
 edge_3_15_0_0 edge_4_1_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_8_0_0
 edge_4_11_0_0 edge_5_1_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_7_1_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_8_0_0 edge_7_9_0_0
 edge_7_10_0_0 edge_7_11_0_0 edge_7_14_0_0 edge_7_15_0_0 edge_8_1_0_0
 edge_8_3_0_0 edge_8_4_0_0 edge_8_5_0_0 edge_8_7_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_8_11_0_0 edge_8_14_0_0 edge_8_15_0_0 edge_9_3_0_0
 edge_9_7_0_0 edge_9_8_0_0 edge_9_11_0_0 edge_9_15_0_0 edge_10_11_0_0
 edge_10_14_0_0 edge_11_1_0_0 edge_11_3_0_0 edge_11_4_0_0 edge_11_5_0_0
 edge_11_7_0_0 edge_11_8_0_0 edge_11_9_0_0 edge_11_10_0_0 edge_11_14_0_0
 edge_11_15_0_0 edge_14_1_0_0 edge_14_3_0_0 edge_14_4_0_0 edge_14_5_0_0
 edge_14_7_0_0 edge_14_8_0_0 edge_14_10_0_0 edge_14_11_0_0 edge_15_9_0_0
 edge_15_11_0_0
End
