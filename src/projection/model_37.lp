Maximize
  - 3.4630473776444 edge_0_2_0_0 - 3.75284 edge_0_3_0_0
   - 2.9253194734424 edge_0_4_0_0 - 1.5634925287783 edge_0_5_0_0
   - 2.35366 edge_0_6_0_0 - 4.0557678098842 edge_0_7_0_0
   + 0.7174886966920401 edge_2_3_0_0 + 0.7874048406019831 edge_2_4_0_0
   + 0.0627732495151533 edge_2_5_0_0 + 0.027909578855704 edge_2_6_0_0
   - 0.507227340905548 edge_2_7_0_0 + 2.328914575305 edge_3_2_0_0
   + 0.6268798871616 edge_3_4_0_0 + 0.0822611462348 edge_3_5_0_0
   - 0.13006 edge_3_6_0_0 - 0.6585605653497399 edge_3_7_0_0
   + 4.606831978598565 edge_4_2_0_0 + 4.0822697550638 edge_4_3_0_0
   + 1.33537647949619 edge_4_5_0_0 + 0.95915782735156 edge_4_6_0_0
   + 1.254329096882522 edge_4_7_0_0 - 0.9571985616664307 edge_5_2_0_0
   - 0.48073817848208 edge_5_3_0_0 + 0.00693031049658235 edge_5_4_0_0
   + 0.23608907746062 edge_5_6_0_0 + 0.3583644387515143 edge_5_7_0_0
   + 1.1799482534055 edge_6_2_0_0 + 0.68597 edge_6_3_0_0
   + 1.364399754408 edge_6_4_0_0 + 2.015739543708 edge_6_5_0_0
   + 1.6869690890362 edge_6_7_0_0 - 1.25833909399525 edge_7_4_0_0
   - 0.7257876973248036 edge_7_5_0_0 + 0.10519 edge_7_6_0_0 + 0 pos_2_0
   + 0 pos_3_0 + 0 pos_4_0 + 0 pos_5_0 + 0 pos_6_0 + 0 pos_7_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0
   + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 = 7
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0
 edge_3_7_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0
 edge_4_7_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 pos_2_0 pos_3_0
 pos_4_0 pos_5_0 pos_6_0 pos_7_0
End
