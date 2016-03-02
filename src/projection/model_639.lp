Maximize
  - 2.3708873208943 edge_0_2_0_0 - 3.4555641810171 edge_0_3_0_0
   - 2.9656722299184 edge_0_5_0_0 - 3.2697577673493 edge_0_6_0_0
   - 4.121223830492299 edge_0_7_0_0 - 3.032569090229 edge_0_8_0_0
   - 3.1359704089776 edge_0_9_0_0 + 2.769051667986479 edge_2_3_0_0
   + 2.503041005335922 edge_2_5_0_0 + 2.473402405384427 edge_2_6_0_0
   + 2.327597612758725 edge_2_7_0_0 + 2.411019602644601 edge_2_8_0_0
   + 1.711491370417145 edge_2_9_0_0 - 0.5738060912988642 edge_3_2_0_0
   + 0.2918862296370348 edge_3_5_0_0 - 0.1404455144654818 edge_3_6_0_0
   - 0.2337300283054702 edge_3_7_0_0 - 0.135812129306829 edge_3_8_0_0
   - 0.1180452021715149 edge_3_9_0_0 + 1.219158753379412 edge_5_2_0_0
   + 1.000553956854715 edge_5_3_0_0 + 2.117418701424038 edge_5_6_0_0
   + 1.551880652017639 edge_5_7_0_0 + 1.761824120811109 edge_5_8_0_0
   + 1.186706811451992 edge_5_9_0_0 - 1.606211877966883 edge_6_2_0_0
   - 0.5799356156164731 edge_6_3_0_0 - 0.7251332310720373 edge_6_5_0_0
   + 0.0367844679085494 edge_6_8_0_0 - 0.2765363000257868 edge_6_9_0_0
   - 1.14703201316446 edge_7_2_0_0 - 0.8591621866795561 edge_7_3_0_0
   - 0.3632027920050418 edge_7_5_0_0 + 0.0380579112783322 edge_7_8_0_0
   + 0.1190326254674976 edge_7_9_0_0 + 0.4575771288729011 edge_8_2_0_0
   + 0.2223734399571514 edge_8_3_0_0 + 1.5837316508253 edge_8_5_0_0
   + 1.197587172958114 edge_8_6_0_0 + 0.5653911026174984 edge_8_7_0_0
   + 1.361511411303194 edge_8_9_0_0 - 2.141099982891885 edge_9_2_0_0
   - 1.192138726849132 edge_9_3_0_0 - 1.66389364460243 edge_9_5_0_0
   - 0.9091458092264569 edge_9_6_0_0 - 1.104993755565209 edge_9_7_0_0
   - 0.4763568065140774 edge_9_8_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_5_0_0
   - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_9_0_0
   + flow_3_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0
   + flow_8_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0 + edge_6_3_0_0
   + edge_7_3_0_0 + edge_8_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0
   + flow_8_3_0_0 + flow_9_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_6_0_0 - flow_5_7_0_0
   - flow_5_8_0_0 - flow_5_9_0_0 + flow_6_5_0_0 + flow_7_5_0_0
   + flow_8_5_0_0 + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_5_6_0_0
   + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_5_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0
   + edge_8_7_0_0 + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_5_7_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_5_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 + flow_8_7_0_0 + flow_9_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_2_0_0
   - flow_8_3_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0
   - flow_8_9_0_0 + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0
   + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_5_0_0 - flow_9_6_0_0
   - flow_9_7_0_0 - flow_9_8_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0 = 9
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
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
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_0_9_0_0 edge_2_3_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_3_2_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_5_0_0 edge_6_8_0_0 edge_6_9_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_5_0_0 edge_7_8_0_0 edge_7_9_0_0
 edge_8_2_0_0 edge_8_3_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0
 edge_8_9_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_8_0_0
End
