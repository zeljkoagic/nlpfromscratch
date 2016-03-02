Maximize
  - 5.0377539136995 edge_0_1_0_0 - 1.211818300608 edge_0_2_0_0
   - 4.532955942924199 edge_0_3_0_0 - 2.1543097095632 edge_0_4_0_0
   - 1.696126394 edge_0_10_0_0 - 2.482212763134 edge_0_11_0_0
   - 4.685602737304499 edge_0_12_0_0 - 0.0381418040112882 edge_1_2_0_0
   - 0.1840070844965865 edge_1_3_0_0 - 0.4429718971201237 edge_1_10_0_0
   - 0.6102405424674396 edge_1_11_0_0 - 0.0904654447448833 edge_1_12_0_0
   - 0.0534377606234538 edge_2_1_0_0 + 0.5435465535201752 edge_2_3_0_0
   + 0.054066616404457 edge_2_4_0_0 - 0.5814081937644635 edge_2_7_0_0
   + 0.2071153270024814 edge_2_10_0_0 - 0.0423054861806972 edge_2_11_0_0
   + 2.014852944399849 edge_2_12_0_0 - 1.034054087506366 edge_3_1_0_0
   - 0.6908225166315163 edge_3_2_0_0 - 0.1135432843708446 edge_3_10_0_0
   - 0.4078661862664726 edge_3_11_0_0 - 0.3963112858200148 edge_3_12_0_0
   - 0.0150042923977175 edge_4_2_0_0 + 0.4489421967765732 edge_4_3_0_0
   + 0.5321841868085292 edge_4_7_0_0 + 0.1341287234826866 edge_4_9_0_0
   + 1.269892173371454 edge_4_10_0_0 + 0.86565078784072 edge_4_11_0_0
   + 1.595768498410248 edge_4_12_0_0 + 0.0992153766245343 edge_7_3_0_0
   + 1.350704964296781 edge_7_6_0_0 + 0.3405428422147569 edge_7_9_0_0
   + 0.1423887108344926 edge_7_10_0_0 + 0.0832011816024114 edge_7_11_0_0
   + 0.3520245648743137 edge_7_12_0_0 - 0.1672749347044786 edge_9_7_0_0
   + 0.1376950495716823 edge_9_11_0_0 - 0.4063461251867824 edge_10_1_0_0
   - 0.2168433533610743 edge_10_2_0_0 + 0.3883471572135806 edge_10_3_0_0
   + 0.1171387112399704 edge_10_4_0_0 + 0.3401934992717613 edge_10_6_0_0
   + 1.690057977169921 edge_10_7_0_0 + 1.962805713393861 edge_10_9_0_0
   + 0.3806499131798275 edge_10_10_0_0 + 0.7804063964965701 edge_10_11_0_0
   + 1.264101291286324 edge_10_12_0_0 - 0.2721233582757824 edge_11_1_0_0
   - 0.5215389242348647 edge_11_2_0_0 + 0.1156133668345919 edge_11_3_0_0
   - 0.3853354460541525 edge_11_7_0_0 - 0.1277884899550808 edge_11_9_0_0
   + 0.2985853020201188 edge_11_10_0_0 + 0.0896875138370981 edge_11_12_0_0
   - 1.351674519665247 edge_12_1_0_0 - 0.4468063179768136 edge_12_3_0_0
   - 0.3221802530983336 edge_12_10_0_0 - 0.1973992397851041 edge_12_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_10_1_0_0
   + edge_11_1_0_0 + edge_12_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_10_0_0 - flow_1_11_0_0 - flow_1_12_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_10_1_0_0 + flow_11_1_0_0 + flow_12_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_7_0_0 - flow_2_10_0_0
   - flow_2_11_0_0 - flow_2_12_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_10_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_7_3_0_0 + edge_10_3_0_0 + edge_11_3_0_0 + edge_12_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_10_0_0 - flow_3_11_0_0
   - flow_3_12_0_0 + flow_4_3_0_0 + flow_7_3_0_0 + flow_10_3_0_0
   + flow_11_3_0_0 + flow_12_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 - flow_4_2_0_0
   - flow_4_3_0_0 - flow_4_7_0_0 - flow_4_9_0_0 - flow_4_10_0_0
   - flow_4_11_0_0 - flow_4_12_0_0 + flow_10_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_7_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_7_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_2_7_0_0 + edge_4_7_0_0 + edge_9_7_0_0 + edge_10_7_0_0
   + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_2_7_0_0 + flow_4_7_0_0 - flow_7_3_0_0
   - flow_7_6_0_0 - flow_7_9_0_0 - flow_7_10_0_0 - flow_7_11_0_0
   - flow_7_12_0_0 + flow_9_7_0_0 + flow_10_7_0_0 + flow_11_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_4_9_0_0 + edge_7_9_0_0 + edge_10_9_0_0 + edge_11_9_0_0
   = 1
 consume_one_flow_unit_9: flow_4_9_0_0 + flow_7_9_0_0 - flow_9_7_0_0
   - flow_9_11_0_0 + flow_10_9_0_0 + flow_11_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_7_10_0_0 + edge_10_10_0_0
   + edge_11_10_0_0 + edge_12_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_7_10_0_0 - flow_10_1_0_0
   - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_6_0_0
   - flow_10_7_0_0 - flow_10_9_0_0 - flow_10_11_0_0 - flow_10_12_0_0
   + flow_11_10_0_0 + flow_12_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_4_11_0_0 + edge_7_11_0_0 + edge_9_11_0_0
   + edge_10_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_4_11_0_0 + flow_7_11_0_0 + flow_9_11_0_0
   + flow_10_11_0_0 - flow_11_1_0_0 - flow_11_2_0_0 - flow_11_3_0_0
   - flow_11_7_0_0 - flow_11_9_0_0 - flow_11_10_0_0 - flow_11_12_0_0
   + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_2_12_0_0
   + edge_3_12_0_0 + edge_4_12_0_0 + edge_7_12_0_0 + edge_10_12_0_0
   + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_2_12_0_0
   + flow_3_12_0_0 + flow_4_12_0_0 + flow_7_12_0_0 + flow_10_12_0_0
   + flow_11_12_0_0 - flow_12_1_0_0 - flow_12_3_0_0 - flow_12_10_0_0
   - flow_12_11_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_10_0_0 + flow_0_11_0_0 + flow_0_12_0_0 = 12
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_10_0_0: flow_10_10_0_0 - 1000 edge_10_10_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_10_0_0
 edge_0_11_0_0 edge_0_12_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_10_0_0
 edge_1_11_0_0 edge_1_12_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_7_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_2_12_0_0 edge_3_1_0_0
 edge_3_2_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_3_12_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_7_0_0 edge_4_9_0_0 edge_4_10_0_0 edge_4_11_0_0
 edge_4_12_0_0 edge_7_3_0_0 edge_7_6_0_0 edge_7_9_0_0 edge_7_10_0_0
 edge_7_11_0_0 edge_7_12_0_0 edge_9_7_0_0 edge_9_11_0_0 edge_10_1_0_0
 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_6_0_0 edge_10_7_0_0
 edge_10_9_0_0 edge_10_10_0_0 edge_10_11_0_0 edge_10_12_0_0 edge_11_1_0_0
 edge_11_2_0_0 edge_11_3_0_0 edge_11_7_0_0 edge_11_9_0_0 edge_11_10_0_0
 edge_11_12_0_0 edge_12_1_0_0 edge_12_3_0_0 edge_12_10_0_0 edge_12_11_0_0
End
