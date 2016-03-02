Maximize
  - 4.2341395997056 edge_0_2_0_0 - 2.7668428335732 edge_0_3_0_0
   - 5.2333086643344 edge_0_4_0_0 - 2.371044526955 edge_0_5_0_0
   - 3.0134262234698 edge_0_6_0_0 - 2.6059073680309 edge_0_9_0_0
   - 3.71254 edge_0_10_0_0 + 0.9165939367996134 edge_2_3_0_0
   + 0.3149486691232417 edge_2_4_0_0 + 0.0480911765980963 edge_2_5_0_0
   - 0.4181302171917649 edge_2_6_0_0 - 1.143550254365877 edge_2_9_0_0
   - 0.5978022042676602 edge_2_10_0_0 + 0.0181933872834 edge_2_11_0_0
   + 2.974192466578391 edge_3_2_0_0 + 3.169698204668932 edge_3_4_0_0
   + 0.3327307123051511 edge_3_5_0_0 + 0.1833806163595748 edge_3_6_0_0
   - 0.0432719413189978 edge_3_7_0_0 + 0.3854616131347158 edge_3_9_0_0
   + 0.4160317994026567 edge_3_10_0_0 + 1.3058640844089 edge_3_11_0_0
   - 0.4702823979022368 edge_4_3_0_0 - 0.8892758348213535 edge_4_5_0_0
   - 1.221150841757793 edge_4_6_0_0 - 1.419755514980397 edge_4_9_0_0
   - 0.8597925431286154 edge_4_11_0_0 + 0.8507302933066048 edge_5_2_0_0
   - 0.275159314008601 edge_5_3_0_0 + 0.6572226656039442 edge_5_4_0_0
   + 1.35940392459995 edge_5_6_0_0 + 0.6753830609897421 edge_5_7_0_0
   + 0.7163086326298104 edge_5_9_0_0 + 0.7100519538699199 edge_5_10_0_0
   + 1.979443254801 edge_5_11_0_0 - 0.8013787183519687 edge_6_2_0_0
   - 0.8563548552845325 edge_6_3_0_0 - 0.9388663782781785 edge_6_4_0_0
   + 0.4662866419045966 edge_6_5_0_0 + 0.1680147362880708 edge_6_7_0_0
   + 1.927277675854381 edge_6_9_0_0 + 1.0450188861543 edge_6_10_0_0
   + 1.2737551267879 edge_6_11_0_0 - 1.14639929034442 edge_7_3_0_0
   - 0.3681721043378088 edge_7_6_0_0 + 0.4459808370597974 edge_7_10_0_0
   - 0.63394976906324 edge_7_11_0_0 + 0.238776920258427 edge_9_2_0_0
   - 0.5501551671022691 edge_9_3_0_0 - 0.786556241245074 edge_9_4_0_0
   + 0.4706761880833299 edge_9_5_0_0 + 0.904396530280546 edge_9_6_0_0
   + 3.394400440562693 edge_9_7_0_0 + 3.1656568026834 edge_9_10_0_0
   + 0.7498192426818001 edge_9_11_0_0 - 0.09834227050158 edge_10_2_0_0
   - 0.3356899833812659 edge_10_3_0_0 - 0.9206203491428 edge_10_4_0_0
   + 0.53971974343192 edge_10_5_0_0 + 0.5206066099452801 edge_10_6_0_0
   + 1.596342159722155 edge_10_7_0_0 + 1.5535884308741 edge_10_9_0_0
   + 0.824286 edge_10_11_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0 + edge_6_2_0_0
   + edge_9_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_9_0_0 - flow_2_10_0_0
   - flow_2_11_0_0 + flow_3_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0
   + flow_9_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0
   - flow_3_9_0_0 - flow_3_10_0_0 - flow_3_11_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0
   + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_9_0_0
   - flow_4_11_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_9_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_9_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_9_0_0 - flow_5_10_0_0
   - flow_5_11_0_0 + flow_6_5_0_0 + flow_9_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_9_0_0
   - flow_6_10_0_0 - flow_6_11_0_0 + flow_7_6_0_0 + flow_9_6_0_0
   + flow_10_6_0_0 = 1
 one_parent_7: edge_3_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_9_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_3_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_3_0_0 - flow_7_6_0_0 - flow_7_10_0_0 - flow_7_11_0_0
   + flow_9_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 - flow_9_2_0_0
   - flow_9_3_0_0 - flow_9_4_0_0 - flow_9_5_0_0 - flow_9_6_0_0
   - flow_9_7_0_0 - flow_9_10_0_0 - flow_9_11_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_9_10_0_0
   - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_5_0_0
   - flow_10_6_0_0 - flow_10_7_0_0 - flow_10_9_0_0 - flow_10_11_0_0 = 1
 one_parent_11: edge_2_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_5_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0 + edge_9_11_0_0
   + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_2_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_5_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0 + flow_9_11_0_0
   + flow_10_11_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_9_0_0 + flow_0_10_0_0 = 11
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0
 edge_2_6_0_0 edge_2_9_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_9_0_0
 edge_3_10_0_0 edge_3_11_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0
 edge_4_9_0_0 edge_4_11_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_6_0_0 edge_5_7_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_7_3_0_0 edge_7_6_0_0
 edge_7_10_0_0 edge_7_11_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0
 edge_9_5_0_0 edge_9_6_0_0 edge_9_7_0_0 edge_9_10_0_0 edge_9_11_0_0
 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_5_0_0 edge_10_6_0_0
 edge_10_7_0_0 edge_10_9_0_0 edge_10_11_0_0
End
