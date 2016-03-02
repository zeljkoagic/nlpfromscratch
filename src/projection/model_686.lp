Maximize
  - 2.8037362231242 edge_0_1_0_0 - 2.7909332551044 edge_0_2_0_0
   - 3.3150437305719 edge_0_3_0_0 - 1.847832944928 edge_0_4_0_0
   - 2.6543771357346 edge_0_7_0_0 - 3.3452587957064 edge_0_8_0_0
   - 4.1509068784404 edge_0_9_0_0 - 3.161699620596 edge_0_11_0_0
   - 4.0600378075784 edge_0_12_0_0 + 1.024081549677318 edge_1_2_0_0
   + 0.6992215420333047 edge_1_3_0_0 + 0.1847558800307321 edge_1_4_0_0
   - 0.426770784466958 edge_1_7_0_0 - 0.3539425343746817 edge_1_8_0_0
   - 0.0884581983276238 edge_1_9_0_0 - 0.5909245604671043 edge_1_11_0_0
   - 0.2233416735742711 edge_1_12_0_0 + 2.425559852984566 edge_2_1_0_0
   + 0.3687157340832389 edge_2_3_0_0 + 0.6524243788649637 edge_2_4_0_0
   - 0.2626858022562915 edge_2_7_0_0 + 0.419418746085117 edge_2_8_0_0
   + 0.2949914732558064 edge_2_9_0_0 - 0.0174060601190865 edge_2_11_0_0
   + 1.119478333934773 edge_2_12_0_0 + 0.1563896053155347 edge_3_1_0_0
   + 0.303271285313185 edge_3_2_0_0 + 0.6929415641254046 edge_3_4_0_0
   - 0.1415365567066768 edge_3_7_0_0 - 0.1001647050618632 edge_3_8_0_0
   + 0.3350135570115569 edge_3_9_0_0 - 0.4085612700172417 edge_3_11_0_0
   + 0.0623154054959279 edge_3_12_0_0 + 0.8227419350455371 edge_4_1_0_0
   + 0.9652782242302735 edge_4_2_0_0 + 2.404774553580342 edge_4_3_0_0
   + 0.3492359763579125 edge_4_7_0_0 + 1.498885105111168 edge_4_8_0_0
   + 0.9154611678005801 edge_4_9_0_0 + 0.9486052609413551 edge_4_11_0_0
   + 1.158599922417704 edge_4_12_0_0 - 0.5160825387914996 edge_7_2_0_0
   + 0.0314852596242031 edge_7_4_0_0 + 0.3319099158227273 edge_7_8_0_0
   - 0.00658105580725542 edge_7_9_0_0 - 0.3493122431592858 edge_7_11_0_0
   - 0.1613891721690399 edge_7_12_0_0 - 0.1553146322870922 edge_8_1_0_0
   - 0.4649084040641343 edge_8_2_0_0 + 0.293993052582783 edge_8_3_0_0
   + 0.2306124324724944 edge_8_4_0_0 + 2.487294091906204 edge_8_7_0_0
   + 2.778299547186403 edge_8_9_0_0 + 2.728328690401718 edge_8_11_0_0
   + 1.032219071002201 edge_8_12_0_0 - 2.051676498740753 edge_9_2_0_0
   - 1.045142804716178 edge_9_4_0_0 - 0.6094927243182888 edge_9_7_0_0
   - 0.3717176480766185 edge_9_8_0_0 - 0.0180253420952587 edge_9_11_0_0
   + 0.0313507299731807 edge_9_12_0_0 - 1.511722835579238 edge_11_2_0_0
   - 0.7378444649886537 edge_11_4_0_0 + 0.1922287232646904 edge_11_7_0_0
   - 0.5204157502003425 edge_11_8_0_0 + 0.5855814284494801 edge_11_9_0_0
   + 0.6720105564727835 edge_11_12_0_0 - 2.827857405034777 edge_12_2_0_0
   - 1.74599433540055 edge_12_4_0_0 - 1.587356166893207 edge_12_7_0_0
   - 1.80227837794835 edge_12_8_0_0 - 1.303213655922046 edge_12_9_0_0
   - 0.9588543671557621 edge_12_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_8_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_7_0_0 - flow_1_8_0_0 - flow_1_9_0_0
   - flow_1_11_0_0 - flow_1_12_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_8_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 + edge_11_2_0_0
   + edge_12_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_7_0_0 - flow_2_8_0_0
   - flow_2_9_0_0 - flow_2_11_0_0 - flow_2_12_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 + flow_9_2_0_0
   + flow_11_2_0_0 + flow_12_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_7_0_0
   - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_11_0_0 - flow_3_12_0_0
   + flow_4_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_7_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0 + edge_11_4_0_0
   + edge_12_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_11_0_0
   - flow_4_12_0_0 + flow_7_4_0_0 + flow_8_4_0_0 + flow_9_4_0_0
   + flow_11_4_0_0 + flow_12_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0 + edge_11_7_0_0
   + edge_12_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 - flow_7_2_0_0 - flow_7_4_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_11_0_0 - flow_7_12_0_0
   + flow_8_7_0_0 + flow_9_7_0_0 + flow_11_7_0_0 + flow_12_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 + edge_11_8_0_0
   + edge_12_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_7_8_0_0 - flow_8_1_0_0
   - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_7_0_0
   - flow_8_9_0_0 - flow_8_11_0_0 - flow_8_12_0_0 + flow_9_8_0_0
   + flow_11_8_0_0 + flow_12_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 + edge_11_9_0_0
   + edge_12_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_2_0_0 - flow_9_4_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_11_0_0 - flow_9_12_0_0 + flow_11_9_0_0 + flow_12_9_0_0 = 1
 one_parent_10: = 1
 consume_one_flow_unit_10: = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_4_11_0_0 + edge_7_11_0_0 + edge_8_11_0_0
   + edge_9_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_4_11_0_0 + flow_7_11_0_0 + flow_8_11_0_0
   + flow_9_11_0_0 - flow_11_2_0_0 - flow_11_4_0_0 - flow_11_7_0_0
   - flow_11_8_0_0 - flow_11_9_0_0 - flow_11_12_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_2_12_0_0
   + edge_3_12_0_0 + edge_4_12_0_0 + edge_7_12_0_0 + edge_8_12_0_0
   + edge_9_12_0_0 + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_2_12_0_0
   + flow_3_12_0_0 + flow_4_12_0_0 + flow_7_12_0_0 + flow_8_12_0_0
   + flow_9_12_0_0 + flow_11_12_0_0 - flow_12_2_0_0 - flow_12_4_0_0
   - flow_12_7_0_0 - flow_12_8_0_0 - flow_12_9_0_0 - flow_12_11_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_11_0_0
   + flow_0_12_0_0 = 12
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_2_0_0: flow_12_2_0_0 - 1000 edge_12_2_0_0 <= 0
 inactive_no_flow_12_4_0_0: flow_12_4_0_0 - 1000 edge_12_4_0_0 <= 0
 inactive_no_flow_12_7_0_0: flow_12_7_0_0 - 1000 edge_12_7_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_7_0_0
 edge_0_8_0_0 edge_0_9_0_0 edge_0_11_0_0 edge_0_12_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_1_9_0_0
 edge_1_11_0_0 edge_1_12_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_11_0_0 edge_2_12_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_3_11_0_0 edge_3_12_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_4_11_0_0
 edge_4_12_0_0 edge_7_2_0_0 edge_7_4_0_0 edge_7_8_0_0 edge_7_9_0_0
 edge_7_11_0_0 edge_7_12_0_0 edge_8_1_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_8_11_0_0 edge_8_12_0_0
 edge_9_2_0_0 edge_9_4_0_0 edge_9_7_0_0 edge_9_8_0_0 edge_9_11_0_0
 edge_9_12_0_0 edge_11_2_0_0 edge_11_4_0_0 edge_11_7_0_0 edge_11_8_0_0
 edge_11_9_0_0 edge_11_12_0_0 edge_12_2_0_0 edge_12_4_0_0 edge_12_7_0_0
 edge_12_8_0_0 edge_12_9_0_0 edge_12_11_0_0
End
