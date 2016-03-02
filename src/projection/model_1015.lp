Maximize
  - 2.9445210280992 edge_0_2_0_0 - 3.1020437057574 edge_0_5_0_0
   - 4.0746967354564 edge_0_6_0_0 - 3.6823 edge_0_7_0_0
   - 5.0941802885864 edge_0_8_0_0 - 3.8408795428014 edge_0_9_0_0
   - 3.48744928529 edge_0_10_0_0 - 2.63412 edge_0_11_0_0
   - 5.522 edge_0_12_0_0 + 0.5631428919889719 edge_2_4_0_0
   + 3.216134112988941 edge_2_5_0_0 + 1.408593747511253 edge_2_6_0_0
   + 0.6395619240806401 edge_2_7_0_0 + 1.966733836106261 edge_2_8_0_0
   - 0.878265914343788 edge_2_9_0_0 + 0.2085374616252274 edge_2_10_0_0
   - 0.12368402624864 edge_2_11_0_0 + 2.4651029235008 edge_2_12_0_0
   - 0.7182606502785162 edge_4_2_0_0 + 0.00823374761643391 edge_4_6_0_0
   - 0.83384699551644 edge_4_7_0_0 - 1.066291719133542 edge_4_8_0_0
   + 1.488786817258946 edge_5_2_0_0 + 3.598855962845422 edge_5_4_0_0
   + 2.930269078088126 edge_5_6_0_0 + 1.9022344456744 edge_5_7_0_0
   + 1.722636447827601 edge_5_8_0_0 - 0.2118566697757286 edge_5_9_0_0
   + 0.4840945781761246 edge_5_10_0_0 + 0.26146017948056 edge_5_11_0_0
   + 1.6183162095636 edge_5_12_0_0 + 0.2284450691219043 edge_6_2_0_0
   + 1.508486935550065 edge_6_4_0_0 + 1.182214961574599 edge_6_5_0_0
   + 1.285151013546 edge_6_7_0_0 + 1.153860129998217 edge_6_8_0_0
   + 0.1017833064718354 edge_6_9_0_0 + 0.3849995303996137 edge_6_10_0_0
   + 0.29235723966852 edge_6_11_0_0 + 0.9418344556056 edge_6_12_0_0
   + 1.1949995204224 edge_7_2_0_0 + 1.1246984086156 edge_7_4_0_0
   + 0.52114968820978 edge_7_5_0_0 + 1.297643247046 edge_7_6_0_0
   + 2.391640505144 edge_7_8_0_0 + 0.60049670079816 edge_7_9_0_0
   + 0.50728361728765 edge_7_10_0_0 + 1.00502 edge_7_11_0_0
   + 0.97914 edge_7_12_0_0 - 0.54517074649444 edge_8_7_0_0
   - 0.3057845955165233 edge_8_9_0_0 - 0.3281093783614525 edge_8_10_0_0
   - 0.202117501124 edge_8_11_0_0 - 0.38268546007428 edge_8_12_0_0
   - 0.2829004791733189 edge_9_2_0_0 - 1.847517926082501 edge_9_5_0_0
   - 1.32432814376017 edge_9_6_0_0 - 0.67983480917334 edge_9_7_0_0
   - 0.3556489777678088 edge_9_8_0_0 - 0.1815607377396846 edge_9_10_0_0
   - 0.14921627695002 edge_9_11_0_0 - 0.4069693898991 edge_9_12_0_0
   - 1.472906559429555 edge_10_2_0_0 - 2.345822907449632 edge_10_5_0_0
   - 2.617708281356082 edge_10_6_0_0 - 2.2043626510635 edge_10_7_0_0
   - 0.580845706114939 edge_10_8_0_0 + 1.276275798831426 edge_10_9_0_0
   + 0.5018014261850501 edge_10_11_0_0 - 0.02737356784939 edge_10_12_0_0
   + 0.79666568201792 edge_11_2_0_0 + 0.9654098941855001 edge_11_4_0_0
   + 0.6817653894626601 edge_11_5_0_0 + 0.66549225928408 edge_11_6_0_0
   + 0.951517 edge_11_7_0_0 + 1.6739085701096 edge_11_8_0_0
   + 2.6498830208292 edge_11_9_0_0 + 2.976058023356 edge_11_10_0_0
   + 2.38867 edge_11_12_0_0 + 0.0842065 edge_12_11_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_4_0_0 - flow_2_5_0_0
   - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_9_0_0
   - flow_2_10_0_0 - flow_2_11_0_0 - flow_2_12_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0
   + flow_10_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: edge_2_4_0_0 + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0
   + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_2_4_0_0 - flow_4_2_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 + flow_5_4_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_6_5_0_0 + edge_7_5_0_0
   + edge_9_5_0_0 + edge_10_5_0_0 + edge_11_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 - flow_5_2_0_0
   - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 - flow_5_10_0_0 - flow_5_11_0_0 - flow_5_12_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 + flow_9_5_0_0 + flow_10_5_0_0
   + flow_11_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0 + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 - flow_6_12_0_0 + flow_7_6_0_0 + flow_9_6_0_0
   + flow_10_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_4_7_0_0 + edge_5_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 + edge_10_7_0_0
   + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_4_7_0_0
   + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0
   - flow_7_10_0_0 - flow_7_11_0_0 - flow_7_12_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 + flow_10_7_0_0 + flow_11_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_4_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0
   + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_4_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_7_0_0
   - flow_8_9_0_0 - flow_8_10_0_0 - flow_8_11_0_0 - flow_8_12_0_0
   + flow_9_8_0_0 + flow_10_8_0_0 + flow_11_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0
   + edge_7_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 + edge_11_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_5_9_0_0
   + flow_6_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0 - flow_9_2_0_0
   - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 - flow_9_11_0_0 - flow_9_12_0_0 + flow_10_9_0_0
   + flow_11_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_5_10_0_0
   + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0 + edge_9_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_5_10_0_0
   + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0 + flow_9_10_0_0
   - flow_10_2_0_0 - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_7_0_0
   - flow_10_8_0_0 - flow_10_9_0_0 - flow_10_11_0_0 - flow_10_12_0_0
   + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_2_11_0_0 + edge_5_11_0_0
   + edge_6_11_0_0 + edge_7_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0
   + edge_10_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_2_11_0_0 + flow_5_11_0_0
   + flow_6_11_0_0 + flow_7_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0
   + flow_10_11_0_0 - flow_11_2_0_0 - flow_11_4_0_0 - flow_11_5_0_0
   - flow_11_6_0_0 - flow_11_7_0_0 - flow_11_8_0_0 - flow_11_9_0_0
   - flow_11_10_0_0 - flow_11_12_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_2_12_0_0 + edge_5_12_0_0
   + edge_6_12_0_0 + edge_7_12_0_0 + edge_8_12_0_0 + edge_9_12_0_0
   + edge_10_12_0_0 + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_2_12_0_0 + flow_5_12_0_0
   + flow_6_12_0_0 + flow_7_12_0_0 + flow_8_12_0_0 + flow_9_12_0_0
   + flow_10_12_0_0 + flow_11_12_0_0 - flow_12_11_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_7_0_0
   + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_10_0_0 + flow_0_11_0_0
   + flow_0_12_0_0 = 12
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_7_0_0 edge_0_8_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_0_12_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0
 edge_2_10_0_0 edge_2_11_0_0 edge_2_12_0_0 edge_4_2_0_0 edge_4_6_0_0
 edge_4_7_0_0 edge_4_8_0_0 edge_5_2_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0
 edge_5_12_0_0 edge_6_2_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_6_12_0_0
 edge_7_2_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0
 edge_7_9_0_0 edge_7_10_0_0 edge_7_11_0_0 edge_7_12_0_0 edge_8_7_0_0
 edge_8_9_0_0 edge_8_10_0_0 edge_8_11_0_0 edge_8_12_0_0 edge_9_2_0_0
 edge_9_5_0_0 edge_9_6_0_0 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0
 edge_9_11_0_0 edge_9_12_0_0 edge_10_2_0_0 edge_10_5_0_0 edge_10_6_0_0
 edge_10_7_0_0 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_10_12_0_0
 edge_11_2_0_0 edge_11_4_0_0 edge_11_5_0_0 edge_11_6_0_0 edge_11_7_0_0
 edge_11_8_0_0 edge_11_9_0_0 edge_11_10_0_0 edge_11_12_0_0 edge_12_11_0_0
End
