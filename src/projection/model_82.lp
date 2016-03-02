Maximize
  - 4.43381489278 edge_0_1_0_0 - 6.0782604495064 edge_0_3_0_0
   - 3.6115576207138 edge_0_4_0_0 - 4.2550161057516 edge_0_5_0_0
   - 4.285087094602599 edge_0_8_0_0 - 4.6917361571544 edge_0_9_0_0
   - 2.6564661421504 edge_0_10_0_0 - 4.5680501424197 edge_0_11_0_0
   + 0.139893017417925 edge_1_3_0_0 + 0.2649155203220336 edge_1_4_0_0
   - 0.1995666210922399 edge_1_5_0_0 + 0.1336961775890068 edge_1_8_0_0
   - 0.7233444154851757 edge_1_10_0_0 - 0.4288795005503482 edge_1_11_0_0
   + 1.256047566845014 edge_3_1_0_0 + 1.607869859262278 edge_3_4_0_0
   + 1.300567967785642 edge_3_5_0_0 - 0.3960567992176381 edge_3_8_0_0
   + 0.1188064732815695 edge_3_9_0_0 + 1.098026965853318 edge_3_10_0_0
   + 0.6921994413213202 edge_3_11_0_0 + 1.648275845021438 edge_4_1_0_0
   + 0.9554011674392069 edge_4_3_0_0 + 1.065896794332467 edge_4_5_0_0
   - 0.1501012046772079 edge_4_8_0_0 - 0.372374408274921 edge_4_9_0_0
   + 0.0286164183893277 edge_4_10_0_0 + 0.5084322065212372 edge_4_11_0_0
   - 0.8207964742350965 edge_5_1_0_0 - 1.227021015717458 edge_5_3_0_0
   - 0.5511694405257641 edge_5_4_0_0 - 0.9474393349762334 edge_5_10_0_0
   - 0.706419756068341 edge_8_1_0_0 - 1.786083667757889 edge_8_3_0_0
   - 1.106259812949945 edge_8_4_0_0 - 0.2899827687018032 edge_8_5_0_0
   - 0.0561771194815722 edge_8_9_0_0 + 0.167224932473628 edge_8_10_0_0
   - 0.0650484632198322 edge_8_11_0_0 + 0.5238301837892544 edge_9_1_0_0
   - 0.4648889539072889 edge_9_3_0_0 - 0.3707418621460045 edge_9_4_0_0
   + 0.2313207615483417 edge_9_5_0_0 + 0.8250680013961146 edge_9_8_0_0
   + 0.8446049015924828 edge_9_10_0_0 + 0.4030517614098549 edge_9_11_0_0
   + 1.76121118175592 edge_10_1_0_0 + 0.9667985506519194 edge_10_3_0_0
   + 0.2939379747434394 edge_10_4_0_0 + 0.8214474743564599 edge_10_5_0_0
   + 3.374799796357268 edge_10_8_0_0 + 3.614513718747968 edge_10_9_0_0
   + 2.891649843806456 edge_10_11_0_0 - 2.109039057683134 edge_11_4_0_0
   - 1.625292111073999 edge_11_8_0_0 - 0.9033590255309129 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0 + edge_5_1_0_0
   + edge_8_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_4_0_0
   - flow_1_5_0_0 - flow_1_8_0_0 - flow_1_10_0_0 - flow_1_11_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 + flow_8_1_0_0
   + flow_9_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_8_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   - flow_3_10_0_0 - flow_3_11_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_8_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_8_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0 + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_3_4_0_0
   - flow_4_1_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 - flow_4_10_0_0 - flow_4_11_0_0 + flow_5_4_0_0
   + flow_8_4_0_0 + flow_9_4_0_0 + flow_10_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_8_5_0_0 + edge_9_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_10_0_0 + flow_8_5_0_0 + flow_9_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_9_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 - flow_8_1_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_5_0_0 - flow_8_9_0_0 - flow_8_10_0_0 - flow_8_11_0_0
   + flow_9_8_0_0 + flow_10_8_0_0 + flow_11_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_3_9_0_0 + flow_4_9_0_0
   + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_3_0_0 - flow_9_4_0_0
   - flow_9_5_0_0 - flow_9_8_0_0 - flow_9_10_0_0 - flow_9_11_0_0
   + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_3_10_0_0
   + edge_4_10_0_0 + edge_5_10_0_0 + edge_8_10_0_0 + edge_9_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_3_10_0_0
   + flow_4_10_0_0 + flow_5_10_0_0 + flow_8_10_0_0 + flow_9_10_0_0
   - flow_10_1_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_5_0_0
   - flow_10_8_0_0 - flow_10_9_0_0 - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_3_11_0_0
   + edge_4_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_3_11_0_0
   + flow_4_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0
   - flow_11_4_0_0 - flow_11_8_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_8_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_1_3_0_0 edge_1_4_0_0
 edge_1_5_0_0 edge_1_8_0_0 edge_1_10_0_0 edge_1_11_0_0 edge_3_1_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0
 edge_3_11_0_0 edge_4_1_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_8_0_0
 edge_4_9_0_0 edge_4_10_0_0 edge_4_11_0_0 edge_5_1_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_10_0_0 edge_8_1_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_9_0_0 edge_8_10_0_0 edge_8_11_0_0 edge_9_1_0_0
 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0 edge_9_8_0_0 edge_9_10_0_0
 edge_9_11_0_0 edge_10_1_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_5_0_0
 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_11_4_0_0 edge_11_8_0_0
 edge_11_10_0_0
End
