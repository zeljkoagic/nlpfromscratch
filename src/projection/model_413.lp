Maximize
  - 1.228851417431 edge_0_1_0_0 - 2.757448488725 edge_0_2_0_0
   - 1.7093909334653 edge_0_3_0_0 - 1.133316565095 edge_0_4_0_0
   - 3.5710096261455 edge_0_5_0_0 - 2.2433966657456 edge_0_6_0_0
   - 3.732671718085 edge_0_8_0_0 - 3.1553652497787 edge_0_9_0_0
   + 0.0437415441986057 edge_1_2_0_0 - 0.0339316633454405 edge_1_3_0_0
   + 0.1858270026697409 edge_1_4_0_0 - 0.2360560547759017 edge_1_5_0_0
   + 0.0153727219126378 edge_1_6_0_0 - 0.000641081 edge_1_7_0_0
   + 9.8944225840740582e-04 edge_1_8_0_0 + 0.7851840116109999 edge_1_9_0_0
   + 1.228333191937561 edge_2_1_0_0 + 1.060837029293025 edge_2_3_0_0
   + 0.7571361841667293 edge_2_4_0_0 + 0.2496003053848639 edge_2_6_0_0
   + 0.3997371712981009 edge_2_7_0_0 + 0.4403822072014514 edge_2_8_0_0
   + 1.235297841267 edge_2_9_0_0 + 0.370182576377552 edge_3_1_0_0
   + 3.010441374725273 edge_3_2_0_0 - 0.6418354825659053 edge_3_4_0_0
   + 3.025286854892785 edge_3_5_0_0 + 0.7397770586585256 edge_3_6_0_0
   + 0.3203805227559421 edge_3_7_0_0 + 0.1526108140563263 edge_3_8_0_0
   + 1.3740555783947 edge_3_9_0_0 + 2.298801182349298 edge_4_1_0_0
   + 1.925781867347281 edge_4_2_0_0 + 1.618838091793854 edge_4_3_0_0
   + 0.4669035069545281 edge_4_5_0_0 + 1.130549773730559 edge_4_6_0_0
   + 0.326555182611448 edge_4_7_0_0 + 1.533292789554167 edge_4_8_0_0
   + 1.821228490038003 edge_4_9_0_0 + 0.4999371224873239 edge_5_1_0_0
   + 0.9863558480199315 edge_5_2_0_0 + 1.797441167525132 edge_5_3_0_0
   + 0.0806079580213543 edge_5_4_0_0 + 2.831387084117731 edge_5_6_0_0
   + 1.2042480372021 edge_5_7_0_0 + 2.295344303924717 edge_5_8_0_0
   + 1.997798742359898 edge_5_9_0_0 + 1.316218722209871 edge_6_1_0_0
   + 0.0690663019127378 edge_6_2_0_0 + 0.1853028436092853 edge_6_3_0_0
   - 0.0650343906201548 edge_6_4_0_0 + 0.7855887007319539 edge_6_5_0_0
   + 0.4917232406033714 edge_6_7_0_0 + 0.6037290717261521 edge_6_8_0_0
   + 0.799913363338919 edge_6_9_0_0 - 0.1584415730876181 edge_7_1_0_0
   - 0.6685661205752894 edge_7_2_0_0 - 0.1495247164136716 edge_7_3_0_0
   - 0.6501271207897283 edge_7_4_0_0 - 0.0760898145284987 edge_7_5_0_0
   + 0.2735382245426751 edge_7_6_0_0 + 0.3568796629988352 edge_7_8_0_0
   - 0.2199330831618293 edge_7_9_0_0 + 0.194878891103874 edge_8_1_0_0
   + 0.8733311240035602 edge_8_2_0_0 + 1.226424050406746 edge_8_3_0_0
   - 0.1294685315132445 edge_8_4_0_0 + 0.8639829451366176 edge_8_5_0_0
   + 2.93035500692084 edge_8_6_0_0 + 3.296032750365765 edge_8_7_0_0
   + 1.6127789355652 edge_8_9_0_0 - 1.331508114942393 edge_9_1_0_0
   - 1.752043168077579 edge_9_2_0_0 - 0.3439171699075318 edge_9_4_0_0
   - 1.343330025224571 edge_9_5_0_0 - 0.6502949219159648 edge_9_6_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0
   + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0
   - flow_1_8_0_0 - flow_1_9_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0
   + flow_8_1_0_0 + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0
   + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_9_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0
   + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0
   + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0
   + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0
   + flow_8_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0
   - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0
   + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0
   + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 + flow_9_6_0_0 = 1
 one_parent_7: edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_1_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0
   - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0 - flow_8_3_0_0
   - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_7_0_0
   - flow_8_9_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0
   + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0
   + flow_7_9_0_0 + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0
   - flow_9_4_0_0 - flow_9_5_0_0 - flow_9_6_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_8_0_0 + flow_0_9_0_0 = 9
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_5_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_8_0_0
 edge_1_9_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0
 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_5_1_0_0
 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0
 edge_5_8_0_0 edge_5_9_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0
 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0
 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0
 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_8_1_0_0 edge_8_2_0_0
 edge_8_3_0_0 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0
 edge_8_9_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_4_0_0 edge_9_5_0_0
 edge_9_6_0_0
End
