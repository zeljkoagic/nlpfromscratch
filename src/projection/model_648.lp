Maximize
  - 3.371947774513 edge_0_2_0_0 - 0.8035477200616 edge_0_3_0_0
   + 0.3494329245990475 edge_1_2_0_0 + 0.0579469966354618 edge_1_3_0_0
   + 0.9817262616152398 edge_2_1_0_0 + 0.1108877833974747 edge_2_3_0_0
   - 0.3506827555999085 edge_2_4_0_0 + 0.3367325751615137 edge_2_5_0_0
   - 0.2834085918720259 edge_2_12_0_0 + 0.3744947833161478 edge_3_1_0_0
   + 0.1365533724221247 edge_3_2_0_0 + 1.205609361021979 edge_3_4_0_0
   + 0.7340455822593601 edge_3_5_0_0 - 0.0425066050539956 edge_3_7_0_0
   + 0.1787048309689792 edge_3_8_0_0 - 0.1251372925036887 edge_3_9_0_0
   + 0.3305043214524192 edge_3_11_0_0 + 0.4072928682626941 edge_3_12_0_0
   - 0.4087280678092156 edge_3_15_0_0 + 0.0915424407323524 edge_3_16_0_0
   + 1.638431959569322 edge_3_17_0_0 - 0.6475215073968785 edge_4_1_0_0
   - 0.9419795955536167 edge_4_2_0_0 + 0.1773733035092576 edge_4_3_0_0
   + 1.068274410713144 edge_4_5_0_0 - 0.3348877816056319 edge_4_7_0_0
   - 0.3058330485288994 edge_4_8_0_0 - 0.3604067758017663 edge_4_11_0_0
   + 1.297191162639128 edge_4_12_0_0 - 0.3116103790540297 edge_4_16_0_0
   + 0.9424524608149557 edge_4_17_0_0 - 0.5558301203968501 edge_5_4_0_0
   + 0.239981452308642 edge_7_8_0_0 - 0.0113915364311988 edge_7_9_0_0
   + 0.5422993849366249 edge_7_11_0_0 - 0.1251202777540829 edge_7_16_0_0
   - 0.4198951493162374 edge_8_1_0_0 + 1.003010630880259 edge_8_7_0_0
   + 0.6702060455493362 edge_8_9_0_0 + 0.7887574477819619 edge_8_11_0_0
   + 0.2233331136524408 edge_8_12_0_0 + 0.0249528136903242 edge_8_15_0_0
   + 0.1608547128031934 edge_8_16_0_0 + 0.385333364550726 edge_8_17_0_0
   + 0.3476561779723281 edge_9_7_0_0 + 0.3992138893651309 edge_9_8_0_0
   + 0.4036691251150123 edge_9_11_0_0 - 0.3873216333875792 edge_9_15_0_0
   - 0.4615124419298888 edge_9_16_0_0 + 0.0555876286256647 edge_11_5_0_0
   + 0.7672886316587894 edge_11_7_0_0 + 0.5106538517501154 edge_11_8_0_0
   + 2.699577320052679 edge_11_9_0_0 + 2.800229370633423 edge_11_12_0_0
   + 0.2175173313644382 edge_11_15_0_0 + 0.938750070488711 edge_11_16_0_0
   + 1.222689543368422 edge_11_17_0_0 - 0.6174555382182955 edge_12_8_0_0
   - 0.7590870386638492 edge_12_9_0_0 - 1.019308834938905 edge_12_11_0_0
   + 0.3530899498358001 edge_15_16_0_0 - 0.0223094201998667 edge_16_7_0_0
   + 0.037442503014652 edge_16_8_0_0 + 0.6253240004131841 edge_16_9_0_0
   + 0.2649018243265009 edge_16_11_0_0 + 0.7284263284423256 edge_16_12_0_0
   + 1.348781129209902 edge_16_15_0_0 + 0.9071839839841669 edge_16_16_0_0
   + 0.8389490811177537 edge_16_17_0_0 - 0.7621465590873856 edge_17_15_0_0
   - 1.064084255947507 edge_17_16_0_0
Subject To
 one_parent_1: edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0 + edge_8_1_0_0
   = 1
 consume_one_flow_unit_1: - flow_1_2_0_0 - flow_1_3_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_12_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_11_0_0
   - flow_3_12_0_0 - flow_3_15_0_0 - flow_3_16_0_0 - flow_3_17_0_0
   + flow_4_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0 = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 - flow_4_1_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_7_0_0
   - flow_4_8_0_0 - flow_4_11_0_0 - flow_4_12_0_0 - flow_4_16_0_0
   - flow_4_17_0_0 + flow_5_4_0_0 = 1
 one_parent_5: edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0 + edge_11_5_0_0
   = 1
 consume_one_flow_unit_5: flow_2_5_0_0 + flow_3_5_0_0 + flow_4_5_0_0
   - flow_5_4_0_0 + flow_11_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_3_7_0_0 + edge_4_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0
   + edge_11_7_0_0 + edge_16_7_0_0 = 1
 consume_one_flow_unit_7: flow_3_7_0_0 + flow_4_7_0_0 - flow_7_8_0_0
   - flow_7_9_0_0 - flow_7_11_0_0 - flow_7_16_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 + flow_11_7_0_0 + flow_16_7_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_4_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0
   + edge_11_8_0_0 + edge_12_8_0_0 + edge_16_8_0_0 = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_4_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_11_0_0
   - flow_8_12_0_0 - flow_8_15_0_0 - flow_8_16_0_0 - flow_8_17_0_0
   + flow_9_8_0_0 + flow_11_8_0_0 + flow_12_8_0_0 + flow_16_8_0_0 = 1
 one_parent_9: edge_3_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 + edge_11_9_0_0
   + edge_12_9_0_0 + edge_16_9_0_0 = 1
 consume_one_flow_unit_9: flow_3_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_11_0_0 - flow_9_15_0_0
   - flow_9_16_0_0 + flow_11_9_0_0 + flow_12_9_0_0 + flow_16_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_3_11_0_0 + edge_4_11_0_0 + edge_7_11_0_0
   + edge_8_11_0_0 + edge_9_11_0_0 + edge_12_11_0_0 + edge_16_11_0_0 = 1
 consume_one_flow_unit_11: flow_3_11_0_0 + flow_4_11_0_0 + flow_7_11_0_0
   + flow_8_11_0_0 + flow_9_11_0_0 - flow_11_5_0_0 - flow_11_7_0_0
   - flow_11_8_0_0 - flow_11_9_0_0 - flow_11_12_0_0 - flow_11_15_0_0
   - flow_11_16_0_0 - flow_11_17_0_0 + flow_12_11_0_0 + flow_16_11_0_0 = 1
 one_parent_12: edge_2_12_0_0 + edge_3_12_0_0 + edge_4_12_0_0
   + edge_8_12_0_0 + edge_11_12_0_0 + edge_16_12_0_0 = 1
 consume_one_flow_unit_12: flow_2_12_0_0 + flow_3_12_0_0 + flow_4_12_0_0
   + flow_8_12_0_0 + flow_11_12_0_0 - flow_12_8_0_0 - flow_12_9_0_0
   - flow_12_11_0_0 + flow_16_12_0_0 = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: = 1
 consume_one_flow_unit_14: = 1
 one_parent_15: edge_3_15_0_0 + edge_8_15_0_0 + edge_9_15_0_0
   + edge_11_15_0_0 + edge_16_15_0_0 + edge_17_15_0_0 = 1
 consume_one_flow_unit_15: flow_3_15_0_0 + flow_8_15_0_0 + flow_9_15_0_0
   + flow_11_15_0_0 - flow_15_16_0_0 + flow_16_15_0_0 + flow_17_15_0_0 = 1
 one_parent_16: edge_3_16_0_0 + edge_4_16_0_0 + edge_7_16_0_0
   + edge_8_16_0_0 + edge_9_16_0_0 + edge_11_16_0_0 + edge_15_16_0_0
   + edge_16_16_0_0 + edge_17_16_0_0 = 1
 consume_one_flow_unit_16: flow_3_16_0_0 + flow_4_16_0_0 + flow_7_16_0_0
   + flow_8_16_0_0 + flow_9_16_0_0 + flow_11_16_0_0 + flow_15_16_0_0
   - flow_16_7_0_0 - flow_16_8_0_0 - flow_16_9_0_0 - flow_16_11_0_0
   - flow_16_12_0_0 - flow_16_15_0_0 - flow_16_17_0_0 + flow_17_16_0_0 = 1
 one_parent_17: edge_3_17_0_0 + edge_4_17_0_0 + edge_8_17_0_0
   + edge_11_17_0_0 + edge_16_17_0_0 = 1
 consume_one_flow_unit_17: flow_3_17_0_0 + flow_4_17_0_0 + flow_8_17_0_0
   + flow_11_17_0_0 + flow_16_17_0_0 - flow_17_15_0_0 - flow_17_16_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 = 17
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_3_15_0_0: flow_3_15_0_0 - 1000 edge_3_15_0_0 <= 0
 inactive_no_flow_3_16_0_0: flow_3_16_0_0 - 1000 edge_3_16_0_0 <= 0
 inactive_no_flow_3_17_0_0: flow_3_17_0_0 - 1000 edge_3_17_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_4_16_0_0: flow_4_16_0_0 - 1000 edge_4_16_0_0 <= 0
 inactive_no_flow_4_17_0_0: flow_4_17_0_0 - 1000 edge_4_17_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_16_0_0: flow_7_16_0_0 - 1000 edge_7_16_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_8_15_0_0: flow_8_15_0_0 - 1000 edge_8_15_0_0 <= 0
 inactive_no_flow_8_16_0_0: flow_8_16_0_0 - 1000 edge_8_16_0_0 <= 0
 inactive_no_flow_8_17_0_0: flow_8_17_0_0 - 1000 edge_8_17_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_15_0_0: flow_9_15_0_0 - 1000 edge_9_15_0_0 <= 0
 inactive_no_flow_9_16_0_0: flow_9_16_0_0 - 1000 edge_9_16_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_11_15_0_0: flow_11_15_0_0 - 1000 edge_11_15_0_0 <= 0
 inactive_no_flow_11_16_0_0: flow_11_16_0_0 - 1000 edge_11_16_0_0 <= 0
 inactive_no_flow_11_17_0_0: flow_11_17_0_0 - 1000 edge_11_17_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
 inactive_no_flow_15_16_0_0: flow_15_16_0_0 - 1000 edge_15_16_0_0 <= 0
 inactive_no_flow_16_7_0_0: flow_16_7_0_0 - 1000 edge_16_7_0_0 <= 0
 inactive_no_flow_16_8_0_0: flow_16_8_0_0 - 1000 edge_16_8_0_0 <= 0
 inactive_no_flow_16_9_0_0: flow_16_9_0_0 - 1000 edge_16_9_0_0 <= 0
 inactive_no_flow_16_11_0_0: flow_16_11_0_0 - 1000 edge_16_11_0_0 <= 0
 inactive_no_flow_16_12_0_0: flow_16_12_0_0 - 1000 edge_16_12_0_0 <= 0
 inactive_no_flow_16_15_0_0: flow_16_15_0_0 - 1000 edge_16_15_0_0 <= 0
 inactive_no_flow_16_16_0_0: flow_16_16_0_0 - 1000 edge_16_16_0_0 <= 0
 inactive_no_flow_16_17_0_0: flow_16_17_0_0 - 1000 edge_16_17_0_0 <= 0
 inactive_no_flow_17_15_0_0: flow_17_15_0_0 - 1000 edge_17_15_0_0 <= 0
 inactive_no_flow_17_16_0_0: flow_17_16_0_0 - 1000 edge_17_16_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_2_1_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_12_0_0 edge_3_1_0_0
 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_3_11_0_0 edge_3_12_0_0 edge_3_15_0_0 edge_3_16_0_0
 edge_3_17_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0
 edge_4_7_0_0 edge_4_8_0_0 edge_4_11_0_0 edge_4_12_0_0 edge_4_16_0_0
 edge_4_17_0_0 edge_5_4_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_7_11_0_0
 edge_7_16_0_0 edge_8_1_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_8_11_0_0
 edge_8_12_0_0 edge_8_15_0_0 edge_8_16_0_0 edge_8_17_0_0 edge_9_7_0_0
 edge_9_8_0_0 edge_9_11_0_0 edge_9_15_0_0 edge_9_16_0_0 edge_11_5_0_0
 edge_11_7_0_0 edge_11_8_0_0 edge_11_9_0_0 edge_11_12_0_0 edge_11_15_0_0
 edge_11_16_0_0 edge_11_17_0_0 edge_12_8_0_0 edge_12_9_0_0 edge_12_11_0_0
 edge_15_16_0_0 edge_16_7_0_0 edge_16_8_0_0 edge_16_9_0_0 edge_16_11_0_0
 edge_16_12_0_0 edge_16_15_0_0 edge_16_16_0_0 edge_16_17_0_0 edge_17_15_0_0
 edge_17_16_0_0
End
