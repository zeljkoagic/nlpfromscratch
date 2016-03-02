Maximize
  - 2.9773904279169 edge_0_2_0_0 - 1.8935086872912 edge_0_3_0_0
   - 2.3295730935825 edge_0_5_0_0 - 2.1085998057375 edge_0_9_0_0
   - 1.0364763735804 edge_0_14_0_0 + 0.2459322909052823 edge_2_3_0_0
   - 0.1401061888415066 edge_2_5_0_0 - 0.2339601662067726 edge_2_7_0_0
   - 0.3706167155449218 edge_2_9_0_0 - 0.374972814961032 edge_2_12_0_0
   - 0.2124912509615805 edge_2_14_0_0 - 0.574094137535647 edge_2_15_0_0
   - 0.1613652946134176 edge_2_16_0_0 + 0.1660530982157263 edge_3_2_0_0
   - 0.4443400152219228 edge_3_5_0_0 - 0.1403730530403815 edge_3_7_0_0
   - 0.3860844355562189 edge_3_16_0_0 + 0.0183531696681634 edge_5_2_0_0
   + 0.5249587485703272 edge_5_3_0_0 + 2.543170771132353 edge_5_7_0_0
   + 1.082792171508671 edge_5_9_0_0 + 1.109697760650279 edge_5_12_0_0
   + 0.2420994178057509 edge_5_14_0_0 + 0.3847936599096132 edge_5_15_0_0
   - 0.1536629897983068 edge_5_16_0_0 - 0.2444500759211901 edge_7_2_0_0
   - 0.1119810151003592 edge_7_3_0_0 + 0.4679777985139676 edge_7_5_0_0
   + 0.8668731020352585 edge_7_9_0_0 + 1.009217896326868 edge_7_12_0_0
   - 0.5276989228300578 edge_7_13_0_0 + 0.126499721699677 edge_7_14_0_0
   + 0.0453685595456717 edge_7_15_0_0 - 0.4299924814981524 edge_7_16_0_0
   - 0.4709816495875567 edge_9_2_0_0 + 0.3932489069813888 edge_9_3_0_0
   + 0.11753024589398 edge_9_5_0_0 + 0.4043888914927827 edge_9_7_0_0
   + 2.603536090025679 edge_9_12_0_0 - 0.7122022902406834 edge_9_13_0_0
   + 0.5021676324171686 edge_9_14_0_0 + 0.8911268660757312 edge_9_15_0_0
   + 0.2076807611542701 edge_9_16_0_0 - 0.5435388894131895 edge_12_2_0_0
   - 0.4215799426228393 edge_12_3_0_0 - 0.6301496602925291 edge_12_5_0_0
   - 0.5589615229008847 edge_12_7_0_0 + 0.3494086028694208 edge_12_9_0_0
   - 0.3217908560729674 edge_12_13_0_0 + 0.1288607212128551 edge_12_14_0_0
   + 0.1855519254664607 edge_12_15_0_0 - 0.5423691182595195 edge_12_16_0_0
   + 0.2404104149041189 edge_13_14_0_0 - 0.4576413691099684 edge_13_15_0_0
   - 0.3625825721666138 edge_14_2_0_0 + 0.0726545271905047 edge_14_3_0_0
   - 0.0991937404607725 edge_14_5_0_0 - 0.2081860914051982 edge_14_7_0_0
   + 0.0440906538638995 edge_14_9_0_0 - 0.00347688944404933 edge_14_12_0_0
   + 1.297996469986038 edge_14_13_0_0 + 0.7924858401907465 edge_14_14_0_0
   + 1.335656432238041 edge_14_15_0_0 + 0.0967002724011281 edge_14_16_0_0
   - 0.7397422364296674 edge_15_12_0_0 + 0.7100604632716463 edge_15_13_0_0
   + 0.5304992466889928 edge_15_14_0_0 - 0.2864389945327493 edge_15_16_0_0
   - 0.8256585456256006 edge_16_14_0_0 - 1.158635454378355 edge_16_15_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0 + edge_7_2_0_0
   + edge_9_2_0_0 + edge_12_2_0_0 + edge_14_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_5_0_0
   - flow_2_7_0_0 - flow_2_9_0_0 - flow_2_12_0_0 - flow_2_14_0_0
   - flow_2_15_0_0 - flow_2_16_0_0 + flow_3_2_0_0 + flow_5_2_0_0
   + flow_7_2_0_0 + flow_9_2_0_0 + flow_12_2_0_0 + flow_14_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0 + edge_7_3_0_0
   + edge_9_3_0_0 + edge_12_3_0_0 + edge_14_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_5_0_0 - flow_3_7_0_0 - flow_3_16_0_0 + flow_5_3_0_0
   + flow_7_3_0_0 + flow_9_3_0_0 + flow_12_3_0_0 + flow_14_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_7_5_0_0
   + edge_9_5_0_0 + edge_12_5_0_0 + edge_14_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_7_0_0 - flow_5_9_0_0
   - flow_5_12_0_0 - flow_5_14_0_0 - flow_5_15_0_0 - flow_5_16_0_0
   + flow_7_5_0_0 + flow_9_5_0_0 + flow_12_5_0_0 + flow_14_5_0_0 = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_2_7_0_0 + edge_3_7_0_0 + edge_5_7_0_0 + edge_9_7_0_0
   + edge_12_7_0_0 + edge_14_7_0_0 = 1
 consume_one_flow_unit_7: flow_2_7_0_0 + flow_3_7_0_0 + flow_5_7_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_5_0_0 - flow_7_9_0_0
   - flow_7_12_0_0 - flow_7_13_0_0 - flow_7_14_0_0 - flow_7_15_0_0
   - flow_7_16_0_0 + flow_9_7_0_0 + flow_12_7_0_0 + flow_14_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_5_9_0_0 + edge_7_9_0_0
   + edge_12_9_0_0 + edge_14_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_5_9_0_0
   + flow_7_9_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_5_0_0
   - flow_9_7_0_0 - flow_9_12_0_0 - flow_9_13_0_0 - flow_9_14_0_0
   - flow_9_15_0_0 - flow_9_16_0_0 + flow_12_9_0_0 + flow_14_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: = 1
 consume_one_flow_unit_11: = 1
 one_parent_12: edge_2_12_0_0 + edge_5_12_0_0 + edge_7_12_0_0
   + edge_9_12_0_0 + edge_14_12_0_0 + edge_15_12_0_0 = 1
 consume_one_flow_unit_12: flow_2_12_0_0 + flow_5_12_0_0 + flow_7_12_0_0
   + flow_9_12_0_0 - flow_12_2_0_0 - flow_12_3_0_0 - flow_12_5_0_0
   - flow_12_7_0_0 - flow_12_9_0_0 - flow_12_13_0_0 - flow_12_14_0_0
   - flow_12_15_0_0 - flow_12_16_0_0 + flow_14_12_0_0 + flow_15_12_0_0 = 1
 one_parent_13: edge_7_13_0_0 + edge_9_13_0_0 + edge_12_13_0_0
   + edge_14_13_0_0 + edge_15_13_0_0 = 1
 consume_one_flow_unit_13: flow_7_13_0_0 + flow_9_13_0_0 + flow_12_13_0_0
   - flow_13_14_0_0 - flow_13_15_0_0 + flow_14_13_0_0 + flow_15_13_0_0 = 1
 one_parent_14: edge_0_14_0_0 + edge_2_14_0_0 + edge_5_14_0_0
   + edge_7_14_0_0 + edge_9_14_0_0 + edge_12_14_0_0 + edge_13_14_0_0
   + edge_14_14_0_0 + edge_15_14_0_0 + edge_16_14_0_0 = 1
 consume_one_flow_unit_14: flow_0_14_0_0 + flow_2_14_0_0 + flow_5_14_0_0
   + flow_7_14_0_0 + flow_9_14_0_0 + flow_12_14_0_0 + flow_13_14_0_0
   - flow_14_2_0_0 - flow_14_3_0_0 - flow_14_5_0_0 - flow_14_7_0_0
   - flow_14_9_0_0 - flow_14_12_0_0 - flow_14_13_0_0 - flow_14_15_0_0
   - flow_14_16_0_0 + flow_15_14_0_0 + flow_16_14_0_0 = 1
 one_parent_15: edge_2_15_0_0 + edge_5_15_0_0 + edge_7_15_0_0
   + edge_9_15_0_0 + edge_12_15_0_0 + edge_13_15_0_0 + edge_14_15_0_0
   + edge_16_15_0_0 = 1
 consume_one_flow_unit_15: flow_2_15_0_0 + flow_5_15_0_0 + flow_7_15_0_0
   + flow_9_15_0_0 + flow_12_15_0_0 + flow_13_15_0_0 + flow_14_15_0_0
   - flow_15_12_0_0 - flow_15_13_0_0 - flow_15_14_0_0 - flow_15_16_0_0
   + flow_16_15_0_0 = 1
 one_parent_16: edge_2_16_0_0 + edge_3_16_0_0 + edge_5_16_0_0
   + edge_7_16_0_0 + edge_9_16_0_0 + edge_12_16_0_0 + edge_14_16_0_0
   + edge_15_16_0_0 = 1
 consume_one_flow_unit_16: flow_2_16_0_0 + flow_3_16_0_0 + flow_5_16_0_0
   + flow_7_16_0_0 + flow_9_16_0_0 + flow_12_16_0_0 + flow_14_16_0_0
   + flow_15_16_0_0 - flow_16_14_0_0 - flow_16_15_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0 + flow_0_9_0_0
   + flow_0_14_0_0 = 16
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_14_0_0: flow_0_14_0_0 - 1000 edge_0_14_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_2_14_0_0: flow_2_14_0_0 - 1000 edge_2_14_0_0 <= 0
 inactive_no_flow_2_15_0_0: flow_2_15_0_0 - 1000 edge_2_15_0_0 <= 0
 inactive_no_flow_2_16_0_0: flow_2_16_0_0 - 1000 edge_2_16_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_16_0_0: flow_3_16_0_0 - 1000 edge_3_16_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_5_14_0_0: flow_5_14_0_0 - 1000 edge_5_14_0_0 <= 0
 inactive_no_flow_5_15_0_0: flow_5_15_0_0 - 1000 edge_5_15_0_0 <= 0
 inactive_no_flow_5_16_0_0: flow_5_16_0_0 - 1000 edge_5_16_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_7_13_0_0: flow_7_13_0_0 - 1000 edge_7_13_0_0 <= 0
 inactive_no_flow_7_14_0_0: flow_7_14_0_0 - 1000 edge_7_14_0_0 <= 0
 inactive_no_flow_7_15_0_0: flow_7_15_0_0 - 1000 edge_7_15_0_0 <= 0
 inactive_no_flow_7_16_0_0: flow_7_16_0_0 - 1000 edge_7_16_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_13_0_0: flow_9_13_0_0 - 1000 edge_9_13_0_0 <= 0
 inactive_no_flow_9_14_0_0: flow_9_14_0_0 - 1000 edge_9_14_0_0 <= 0
 inactive_no_flow_9_15_0_0: flow_9_15_0_0 - 1000 edge_9_15_0_0 <= 0
 inactive_no_flow_9_16_0_0: flow_9_16_0_0 - 1000 edge_9_16_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_3_0_0: flow_12_3_0_0 - 1000 edge_12_3_0_0 <= 0
 inactive_no_flow_12_5_0_0: flow_12_5_0_0 - 1000 edge_12_5_0_0 <= 0
 inactive_no_flow_12_7_0_0: flow_12_7_0_0 - 1000 edge_12_7_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_13_0_0: flow_12_13_0_0 - 1000 edge_12_13_0_0 <= 0
 inactive_no_flow_12_14_0_0: flow_12_14_0_0 - 1000 edge_12_14_0_0 <= 0
 inactive_no_flow_12_15_0_0: flow_12_15_0_0 - 1000 edge_12_15_0_0 <= 0
 inactive_no_flow_12_16_0_0: flow_12_16_0_0 - 1000 edge_12_16_0_0 <= 0
 inactive_no_flow_13_14_0_0: flow_13_14_0_0 - 1000 edge_13_14_0_0 <= 0
 inactive_no_flow_13_15_0_0: flow_13_15_0_0 - 1000 edge_13_15_0_0 <= 0
 inactive_no_flow_14_2_0_0: flow_14_2_0_0 - 1000 edge_14_2_0_0 <= 0
 inactive_no_flow_14_3_0_0: flow_14_3_0_0 - 1000 edge_14_3_0_0 <= 0
 inactive_no_flow_14_5_0_0: flow_14_5_0_0 - 1000 edge_14_5_0_0 <= 0
 inactive_no_flow_14_7_0_0: flow_14_7_0_0 - 1000 edge_14_7_0_0 <= 0
 inactive_no_flow_14_9_0_0: flow_14_9_0_0 - 1000 edge_14_9_0_0 <= 0
 inactive_no_flow_14_12_0_0: flow_14_12_0_0 - 1000 edge_14_12_0_0 <= 0
 inactive_no_flow_14_13_0_0: flow_14_13_0_0 - 1000 edge_14_13_0_0 <= 0
 inactive_no_flow_14_14_0_0: flow_14_14_0_0 - 1000 edge_14_14_0_0 <= 0
 inactive_no_flow_14_15_0_0: flow_14_15_0_0 - 1000 edge_14_15_0_0 <= 0
 inactive_no_flow_14_16_0_0: flow_14_16_0_0 - 1000 edge_14_16_0_0 <= 0
 inactive_no_flow_15_12_0_0: flow_15_12_0_0 - 1000 edge_15_12_0_0 <= 0
 inactive_no_flow_15_13_0_0: flow_15_13_0_0 - 1000 edge_15_13_0_0 <= 0
 inactive_no_flow_15_14_0_0: flow_15_14_0_0 - 1000 edge_15_14_0_0 <= 0
 inactive_no_flow_15_16_0_0: flow_15_16_0_0 - 1000 edge_15_16_0_0 <= 0
 inactive_no_flow_16_14_0_0: flow_16_14_0_0 - 1000 edge_16_14_0_0 <= 0
 inactive_no_flow_16_15_0_0: flow_16_15_0_0 - 1000 edge_16_15_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_9_0_0 edge_0_14_0_0
 edge_2_3_0_0 edge_2_5_0_0 edge_2_7_0_0 edge_2_9_0_0 edge_2_12_0_0
 edge_2_14_0_0 edge_2_15_0_0 edge_2_16_0_0 edge_3_2_0_0 edge_3_5_0_0
 edge_3_7_0_0 edge_3_16_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_7_0_0
 edge_5_9_0_0 edge_5_12_0_0 edge_5_14_0_0 edge_5_15_0_0 edge_5_16_0_0
 edge_7_2_0_0 edge_7_3_0_0 edge_7_5_0_0 edge_7_9_0_0 edge_7_12_0_0
 edge_7_13_0_0 edge_7_14_0_0 edge_7_15_0_0 edge_7_16_0_0 edge_9_2_0_0
 edge_9_3_0_0 edge_9_5_0_0 edge_9_7_0_0 edge_9_12_0_0 edge_9_13_0_0
 edge_9_14_0_0 edge_9_15_0_0 edge_9_16_0_0 edge_12_2_0_0 edge_12_3_0_0
 edge_12_5_0_0 edge_12_7_0_0 edge_12_9_0_0 edge_12_13_0_0 edge_12_14_0_0
 edge_12_15_0_0 edge_12_16_0_0 edge_13_14_0_0 edge_13_15_0_0 edge_14_2_0_0
 edge_14_3_0_0 edge_14_5_0_0 edge_14_7_0_0 edge_14_9_0_0 edge_14_12_0_0
 edge_14_13_0_0 edge_14_14_0_0 edge_14_15_0_0 edge_14_16_0_0 edge_15_12_0_0
 edge_15_13_0_0 edge_15_14_0_0 edge_15_16_0_0 edge_16_14_0_0 edge_16_15_0_0
End
