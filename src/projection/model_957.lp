Maximize
  - 3.609997445218 edge_0_1_0_0 - 3.8177704452846 edge_0_2_0_0
   - 1.9581598825104 edge_0_3_0_0 - 3.9117190611872 edge_0_4_0_0
   - 3.9384516414183 edge_0_7_0_0 - 3.937682782593 edge_0_9_0_0
   - 3.9738043174517 edge_0_10_0_0 - 4.8620267910602 edge_0_11_0_0
   + 1.352016226732586 edge_1_2_0_0 + 1.790725875537241 edge_1_3_0_0
   + 1.118277055735042 edge_1_4_0_0 + 0.2021648435364902 edge_1_7_0_0
   + 0.3482513764725795 edge_1_10_0_0 + 1.286201156160726 edge_1_11_0_0
   + 0.4523386564947014 edge_2_1_0_0 + 0.4450330468500556 edge_2_3_0_0
   + 0.2150801790025402 edge_2_4_0_0 - 0.4313405646710946 edge_2_7_0_0
   - 0.502815653539205 edge_2_9_0_0 - 0.4659939174985775 edge_2_10_0_0
   + 0.368280400865005 edge_2_11_0_0 + 3.176781708089886 edge_3_1_0_0
   + 4.035005682290344 edge_3_2_0_0 + 2.835379149386041 edge_3_4_0_0
   + 0.5295037737568117 edge_3_6_0_0 + 1.939732590193638 edge_3_7_0_0
   - 0.1620279661308915 edge_3_8_0_0 + 1.311518045835313 edge_3_10_0_0
   + 2.534589695849209 edge_3_11_0_0 + 0.9945268592384721 edge_4_1_0_0
   + 1.476699032736747 edge_4_2_0_0 + 1.154371848232686 edge_4_3_0_0
   + 2.003628141076331 edge_4_6_0_0 + 3.245607017522929 edge_4_7_0_0
   + 0.6377146713184622 edge_4_8_0_0 + 1.011610362126467 edge_4_9_0_0
   + 1.74468637104661 edge_4_10_0_0 + 1.719006716692318 edge_4_11_0_0
   + 1.22083764899878 edge_6_7_0_0 + 0.4604060245876156 edge_6_8_0_0
   - 0.1161381226436072 edge_6_10_0_0 - 0.4640663983116945 edge_6_11_0_0
   + 0.4444342334538446 edge_7_1_0_0 + 0.631744332349869 edge_7_2_0_0
   + 0.0595761611559966 edge_7_3_0_0 + 1.337294650801207 edge_7_4_0_0
   + 2.591874227161863 edge_7_6_0_0 + 0.8629583121769659 edge_7_8_0_0
   + 0.4688865102021427 edge_7_9_0_0 + 2.230678422730894 edge_7_10_0_0
   + 0.8430307796103101 edge_7_11_0_0 - 0.2486708236686217 edge_8_6_0_0
   - 0.0631243001840159 edge_8_7_0_0 + 1.447376042809499 edge_8_10_0_0
   + 0.0644820688634296 edge_8_11_0_0 - 0.0549033031531685 edge_9_1_0_0
   - 0.1736002723109087 edge_9_2_0_0 + 0.4841368679111764 edge_9_3_0_0
   - 0.0660933035748272 edge_9_4_0_0 + 1.647085938522404 edge_9_7_0_0
   + 0.3438973937412435 edge_9_8_0_0 + 0.5326888199239125 edge_9_10_0_0
   - 0.0424038251699469 edge_9_11_0_0 + 0.5799103499430101 edge_10_1_0_0
   + 0.3851972213902557 edge_10_2_0_0 - 0.0406222938334181 edge_10_3_0_0
   + 0.4010373302666776 edge_10_4_0_0 + 0.384275348409226 edge_10_6_0_0
   + 2.092702480649699 edge_10_7_0_0 + 2.67204349403374 edge_10_8_0_0
   + 3.403670771651915 edge_10_9_0_0 + 0.5768131405471896 edge_10_11_0_0
   - 1.1641582654017 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_7_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_7_0_0 - flow_1_10_0_0 - flow_1_11_0_0
   + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0 + flow_7_1_0_0
   + flow_9_1_0_0 + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_7_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_7_0_0 - flow_2_9_0_0
   - flow_2_10_0_0 - flow_2_11_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_7_2_0_0 + flow_9_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_7_3_0_0 + edge_9_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_10_0_0 - flow_3_11_0_0
   + flow_4_3_0_0 + flow_7_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_7_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0
   - flow_4_10_0_0 - flow_4_11_0_0 + flow_7_4_0_0 + flow_9_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_3_6_0_0 + edge_4_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_3_6_0_0 + flow_4_6_0_0 - flow_6_7_0_0
   - flow_6_8_0_0 - flow_6_10_0_0 - flow_6_11_0_0 + flow_7_6_0_0
   + flow_8_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0 - flow_7_11_0_0
   + flow_8_7_0_0 + flow_9_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_3_8_0_0 + edge_4_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0
   + edge_9_8_0_0 + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_3_8_0_0 + flow_4_8_0_0 + flow_6_8_0_0
   + flow_7_8_0_0 - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_10_0_0
   - flow_8_11_0_0 + flow_9_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_4_9_0_0 + edge_7_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_4_9_0_0
   + flow_7_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0 - flow_9_3_0_0
   - flow_9_4_0_0 - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_10_0_0
   - flow_9_11_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_3_10_0_0 + edge_4_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_3_10_0_0 + flow_4_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_2_0_0
   - flow_10_3_0_0 - flow_10_4_0_0 - flow_10_6_0_0 - flow_10_7_0_0
   - flow_10_8_0_0 - flow_10_9_0_0 - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_4_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0
   + edge_8_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_4_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0
   + flow_8_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_7_0_0 + flow_0_9_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_7_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_7_0_0 edge_1_10_0_0 edge_1_11_0_0 edge_2_1_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_7_0_0 edge_2_9_0_0 edge_2_10_0_0
 edge_2_11_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_6_0_0
 edge_3_7_0_0 edge_3_8_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0
 edge_4_9_0_0 edge_4_10_0_0 edge_4_11_0_0 edge_6_7_0_0 edge_6_8_0_0
 edge_6_10_0_0 edge_6_11_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0
 edge_7_4_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_7_10_0_0
 edge_7_11_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_10_0_0 edge_8_11_0_0
 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_7_0_0
 edge_9_8_0_0 edge_9_10_0_0 edge_9_11_0_0 edge_10_1_0_0 edge_10_2_0_0
 edge_10_3_0_0 edge_10_4_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_8_0_0
 edge_10_9_0_0 edge_10_11_0_0 edge_11_10_0_0
End
