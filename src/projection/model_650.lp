Maximize
  - 2.905280608224 edge_0_1_0_0 - 2.5065198496088 edge_0_2_0_0
   - 0.84402648771428 edge_0_3_0_0 - 2.6634616965123 edge_0_4_0_0
   + 0.41975139271915 edge_0_5_0_0 - 3.430512624382 edge_0_10_0_0
   - 0.071994111201066 edge_1_2_0_0 - 0.3264989560902067 edge_1_3_0_0
   - 0.0608929708631128 edge_1_4_0_0 - 0.7248852869026845 edge_1_5_0_0
   - 1.268664050575227 edge_1_10_0_0 - 0.3379727371976526 edge_1_11_0_0
   + 0.6126934371971116 edge_2_1_0_0 + 0.9891847924099889 edge_2_3_0_0
   + 0.9612057412087021 edge_2_4_0_0 + 0.5519338870048648 edge_2_5_0_0
   - 0.4428562175414253 edge_2_6_0_0 - 0.6011545399987854 edge_2_9_0_0
   - 0.2437249853765 edge_2_10_0_0 + 0.7549715470168245 edge_2_11_0_0
   + 2.0016249392207 edge_3_1_0_0 + 2.294589005130949 edge_3_2_0_0
   + 1.715772001345117 edge_3_4_0_0 + 1.947914241637084 edge_3_5_0_0
   + 0.182385247780692 edge_3_6_0_0 - 0.4983632805631788 edge_3_9_0_0
   - 0.011028575706551 edge_3_10_0_0 + 2.475980758336969 edge_3_11_0_0
   - 0.4159297960309092 edge_4_1_0_0 - 0.4532311653007463 edge_4_2_0_0
   - 0.4337841409798405 edge_4_3_0_0 - 0.3234294925864005 edge_4_5_0_0
   + 1.395868621686857 edge_5_1_0_0 + 0.4616010336985075 edge_5_2_0_0
   + 0.8837476496533898 edge_5_3_0_0 + 1.238485776195071 edge_5_4_0_0
   + 2.955590432507287 edge_5_6_0_0 + 0.8269221568058892 edge_5_9_0_0
   + 2.389021214767045 edge_5_10_0_0 + 2.515081321413353 edge_5_11_0_0
   - 0.6477123698667782 edge_6_1_0_0 - 0.7160947080474952 edge_6_2_0_0
   - 0.5835835230395718 edge_6_3_0_0 - 0.108420286619013 edge_6_4_0_0
   - 0.4217468636415033 edge_6_5_0_0 + 0.2526162646876755 edge_6_9_0_0
   + 0.7582577069536 edge_6_10_0_0 + 0.8283419300079161 edge_6_11_0_0
   - 0.9519456470053332 edge_9_1_0_0 - 0.9487427940244103 edge_9_2_0_0
   - 1.106928038223301 edge_9_3_0_0 - 0.3922227110306686 edge_9_4_0_0
   + 0.3929577518320023 edge_9_6_0_0 + 0.5034647325288899 edge_9_10_0_0
   + 0.0304218006115508 edge_9_11_0_0 - 0.5628831179999343 edge_10_1_0_0
   - 0.58207196507568 edge_10_2_0_0 - 0.48258956115435 edge_10_3_0_0
   + 0.18444269857683 edge_10_4_0_0 + 0.3891870729263542 edge_10_5_0_0
   + 1.098091909264 edge_10_6_0_0 + 3.410049474033368 edge_10_9_0_0
   + 1.3142092903266 edge_10_11_0_0 - 1.069795751757982 edge_11_6_0_0
   - 0.027709493864692 edge_11_9_0_0 - 0.2071582025782568 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_10_0_0 - flow_1_11_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0
   + flow_6_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 - flow_2_11_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_9_2_0_0
   + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_9_0_0 - flow_3_10_0_0 - flow_3_11_0_0
   + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0 + flow_9_3_0_0
   + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_9_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_4_5_0_0 + edge_6_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0
   - flow_5_3_0_0 - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_9_0_0
   - flow_5_10_0_0 - flow_5_11_0_0 + flow_6_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_2_6_0_0 + edge_3_6_0_0 + edge_5_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_2_6_0_0 + flow_3_6_0_0 + flow_5_6_0_0
   - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_5_0_0 - flow_6_9_0_0 - flow_6_10_0_0 - flow_6_11_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_2_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0
   + edge_10_9_0_0 + edge_11_9_0_0 = 1
 consume_one_flow_unit_9: flow_2_9_0_0 + flow_3_9_0_0 + flow_5_9_0_0
   + flow_6_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0
   - flow_9_4_0_0 - flow_9_6_0_0 - flow_9_10_0_0 - flow_9_11_0_0
   + flow_10_9_0_0 + flow_11_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0 + edge_9_10_0_0
   + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0 + flow_9_10_0_0
   - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_9_0_0 - flow_10_11_0_0
   + flow_11_10_0_0 = 1
 one_parent_11: edge_1_11_0_0 + edge_2_11_0_0 + edge_3_11_0_0
   + edge_5_11_0_0 + edge_6_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_2_11_0_0 + flow_3_11_0_0
   + flow_5_11_0_0 + flow_6_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0
   - flow_11_6_0_0 - flow_11_9_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_10_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_10_0_0 edge_1_2_0_0 edge_1_3_0_0 edge_1_4_0_0 edge_1_5_0_0
 edge_1_10_0_0 edge_1_11_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_9_0_0 edge_2_10_0_0 edge_2_11_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0
 edge_3_9_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_6_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0
 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_9_1_0_0 edge_9_2_0_0
 edge_9_3_0_0 edge_9_4_0_0 edge_9_6_0_0 edge_9_10_0_0 edge_9_11_0_0
 edge_10_1_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_5_0_0
 edge_10_6_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_11_6_0_0 edge_11_9_0_0
 edge_11_10_0_0
End
