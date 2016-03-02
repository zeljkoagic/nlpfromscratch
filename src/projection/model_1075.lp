Maximize
  - 2.5450052520298 edge_0_1_0_0 - 3.5121795785384 edge_0_3_0_0
   - 0.5375312889204 edge_0_4_0_0 - 3.574549739012 edge_0_6_0_0
   - 0.00696617964974234 edge_1_3_0_0 + 0.0823449894619206 edge_1_4_0_0
   - 0.6362305266565167 edge_1_6_0_0 + 0.4231996044022367 edge_1_7_0_0
   - 0.9683222055166212 edge_2_1_0_0 + 0.2681062205075471 edge_2_3_0_0
   + 0.3650166065452788 edge_2_4_0_0 + 0.5586579496809185 edge_2_7_0_0
   + 0.4282528036913174 edge_3_1_0_0 + 3.925192073362894 edge_3_2_0_0
   - 0.012795215035555 edge_3_4_0_0 - 0.163057589915296 edge_3_5_0_0
   + 0.0425220012386742 edge_3_6_0_0 + 0.7746429517784039 edge_3_7_0_0
   + 0.3935043919087465 edge_4_1_0_0 + 0.6062918676333321 edge_4_2_0_0
   + 2.004012719469245 edge_4_3_0_0 + 0.3946648095291367 edge_4_4_0_0
   + 0.3379311824907976 edge_4_5_0_0 + 2.380393420646063 edge_4_6_0_0
   + 3.014277009335461 edge_4_7_0_0 - 0.0848137605965381 edge_5_2_0_0
   - 0.3518368201578144 edge_5_3_0_0 + 0.222159453781818 edge_5_4_0_0
   + 0.00804262407824192 edge_5_6_0_0 + 0.2652099157074684 edge_5_7_0_0
   - 0.00723722284236891 edge_6_1_0_0 + 0.9678151535427254 edge_6_2_0_0
   + 0.2255267095049016 edge_6_3_0_0 + 1.027483211747277 edge_6_4_0_0
   + 1.039423914086864 edge_6_7_0_0 - 1.409192279360682 edge_7_1_0_0
   - 0.4706234124660828 edge_7_6_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_4_0_0
   - flow_1_6_0_0 - flow_1_7_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0 = 1
 one_parent_2: edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   = 1
 consume_one_flow_unit_2: - flow_2_1_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_7_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_6_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_4_4_0_0 + edge_5_4_0_0 + edge_6_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 + flow_5_4_0_0
   + flow_6_4_0_0 = 1
 one_parent_5: edge_3_5_0_0 + edge_4_5_0_0 = 1
 consume_one_flow_unit_5: flow_3_5_0_0 + flow_4_5_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_7_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 = 1
 consume_one_flow_unit_7: flow_1_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_6_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_6_0_0 = 7
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_4_0_0: flow_4_4_0_0 - 1000 edge_4_4_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_2_1_0_0 edge_2_3_0_0
 edge_2_4_0_0 edge_2_7_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_4_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0
 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0
 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_7_0_0
 edge_7_1_0_0 edge_7_6_0_0
End
