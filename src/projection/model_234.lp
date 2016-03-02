Maximize
  - 5.1454543770072 edge_0_1_0_0 - 5.0309815270663 edge_0_2_0_0
   - 4.3037 edge_0_3_0_0 - 4.3670551953644 edge_0_4_0_0
   - 5.0052714159204 edge_0_6_0_0 - 4.06519 edge_0_7_0_0
   - 4.0724634434938 edge_0_8_0_0 - 3.7995594400416 edge_0_11_0_0
   - 5.0603692102784 edge_0_12_0_0 - 0.5026598868127435 edge_1_2_0_0
   - 0.8363584530648001 edge_1_3_0_0 - 0.063534885135322 edge_1_4_0_0
   - 1.6857490909968 edge_1_7_0_0 - 0.2481381667564776 edge_1_8_0_0
   - 2.002458820952012 edge_1_11_0_0 - 0.3119586449850549 edge_1_12_0_0
   - 1.195800322910007 edge_2_1_0_0 + 0.8885885573294999 edge_2_3_0_0
   + 0.2982026037403189 edge_2_4_0_0 - 0.4281263924258315 edge_2_6_0_0
   - 0.41280330082725 edge_2_7_0_0 - 0.2317068061476106 edge_2_8_0_0
   - 0.8994628400896416 edge_2_10_0_0 - 0.8724005681287518 edge_2_11_0_0
   - 0.9344706885308486 edge_2_12_0_0 - 0.1558365480324 edge_3_1_0_0
   + 3.3371220090501 edge_3_2_0_0 + 2.1826377119486 edge_3_4_0_0
   + 0.65353670275968 edge_3_6_0_0 + 2.25589 edge_3_7_0_0
   + 2.1072290301819 edge_3_8_0_0 - 0.41907188672657 edge_3_10_0_0
   + 1.5105004274917 edge_3_11_0_0 + 1.5333474495864 edge_3_12_0_0
   - 1.884333885680803 edge_4_1_0_0 - 0.8836239215245052 edge_4_2_0_0
   - 0.91827336386642 edge_4_3_0_0 - 0.8023525037790186 edge_4_6_0_0
   - 0.49131696967552 edge_4_7_0_0 - 0.5594056753980022 edge_4_8_0_0
   - 0.7911486037189666 edge_4_11_0_0 - 0.7062324607247897 edge_4_12_0_0
   - 1.559847830748987 edge_6_1_0_0 - 0.9996770613600001 edge_6_3_0_0
   + 0.3022333852007491 edge_6_4_0_0 + 0.8595382312131918 edge_6_7_0_0
   + 0.3169100911829212 edge_6_8_0_0 - 0.4869545324611237 edge_6_10_0_0
   - 0.2237247134503685 edge_6_11_0_0 - 1.143373242640096 edge_6_12_0_0
   - 0.2884370302548 edge_7_1_0_0 + 0.7221325385655 edge_7_2_0_0
   + 0.0306344 edge_7_3_0_0 + 1.686896934609532 edge_7_4_0_0
   + 3.51034434824 edge_7_6_0_0 + 2.0084109387792 edge_7_8_0_0
   + 0.5050542911710735 edge_7_10_0_0 + 2.1158972062896 edge_7_11_0_0
   + 1.426138418094 edge_7_12_0_0 - 2.149365049229203 edge_8_1_0_0
   - 1.7198802454515 edge_8_3_0_0 - 1.259673581017234 edge_8_4_0_0
   - 0.8765015042922802 edge_8_6_0_0 - 0.9182001236369101 edge_8_7_0_0
   - 0.8022738560803199 edge_8_10_0_0 - 0.4911381077185023 edge_8_11_0_0
   - 0.6645532855866823 edge_8_12_0_0 - 1.694580620558453 edge_10_1_0_0
   - 0.7624614725801656 edge_10_4_0_0 + 0.2299743883965002 edge_10_8_0_0
   + 1.045688734866504 edge_10_11_0_0 - 0.831330085489144 edge_10_12_0_0
   - 0.1952765948738598 edge_11_1_0_0 - 0.5417806558311469 edge_11_2_0_0
   - 0.69374170620291 edge_11_3_0_0 + 0.371072114917521 edge_11_4_0_0
   + 0.7839480415968212 edge_11_6_0_0 + 0.19207637523539 edge_11_7_0_0
   + 1.381027160182939 edge_11_8_0_0 + 3.360506199581161 edge_11_10_0_0
   + 1.396685022705198 edge_11_12_0_0 - 2.293725308578713 edge_12_1_0_0
   - 1.004349987567352 edge_12_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0 + edge_10_1_0_0
   + edge_11_1_0_0 + edge_12_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_7_0_0 - flow_1_8_0_0 - flow_1_11_0_0
   - flow_1_12_0_0 + flow_2_1_0_0 + flow_3_1_0_0 + flow_4_1_0_0
   + flow_6_1_0_0 + flow_7_1_0_0 + flow_8_1_0_0 + flow_10_1_0_0
   + flow_11_1_0_0 + flow_12_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_7_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_10_0_0 - flow_2_11_0_0 - flow_2_12_0_0
   + flow_3_2_0_0 + flow_4_2_0_0 + flow_7_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_6_0_0
   - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_10_0_0 - flow_3_11_0_0
   - flow_3_12_0_0 + flow_4_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0
   + flow_8_3_0_0 + flow_11_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 + edge_10_4_0_0
   + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_11_0_0
   - flow_4_12_0_0 + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_10_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 - flow_6_1_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_10_0_0 - flow_6_11_0_0
   - flow_6_12_0_0 + flow_7_6_0_0 + flow_8_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_1_0_0
   - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_10_0_0 - flow_7_11_0_0 - flow_7_12_0_0
   + flow_8_7_0_0 + flow_11_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0
   + edge_4_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_10_8_0_0
   + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_2_8_0_0
   + flow_3_8_0_0 + flow_4_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_6_0_0
   - flow_8_7_0_0 - flow_8_10_0_0 - flow_8_11_0_0 - flow_8_12_0_0
   + flow_10_8_0_0 + flow_11_8_0_0 = 1
 one_parent_9: = 1
 consume_one_flow_unit_9: = 1
 one_parent_10: edge_2_10_0_0 + edge_3_10_0_0 + edge_6_10_0_0
   + edge_7_10_0_0 + edge_8_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_3_10_0_0 + flow_6_10_0_0
   + flow_7_10_0_0 + flow_8_10_0_0 - flow_10_1_0_0 - flow_10_4_0_0
   - flow_10_8_0_0 - flow_10_11_0_0 - flow_10_12_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_3_11_0_0 + edge_4_11_0_0 + edge_6_11_0_0 + edge_7_11_0_0
   + edge_8_11_0_0 + edge_10_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_3_11_0_0 + flow_4_11_0_0 + flow_6_11_0_0 + flow_7_11_0_0
   + flow_8_11_0_0 + flow_10_11_0_0 - flow_11_1_0_0 - flow_11_2_0_0
   - flow_11_3_0_0 - flow_11_4_0_0 - flow_11_6_0_0 - flow_11_7_0_0
   - flow_11_8_0_0 - flow_11_10_0_0 - flow_11_12_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_2_12_0_0
   + edge_3_12_0_0 + edge_4_12_0_0 + edge_6_12_0_0 + edge_7_12_0_0
   + edge_8_12_0_0 + edge_10_12_0_0 + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_2_12_0_0
   + flow_3_12_0_0 + flow_4_12_0_0 + flow_6_12_0_0 + flow_7_12_0_0
   + flow_8_12_0_0 + flow_10_12_0_0 + flow_11_12_0_0 - flow_12_1_0_0
   - flow_12_11_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_11_0_0
   + flow_0_12_0_0 = 12
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_7_12_0_0: flow_7_12_0_0 - 1000 edge_7_12_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_0_11_0_0 edge_0_12_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_7_0_0 edge_1_8_0_0 edge_1_11_0_0
 edge_1_12_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_2_10_0_0 edge_2_11_0_0 edge_2_12_0_0
 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_6_0_0 edge_3_7_0_0
 edge_3_8_0_0 edge_3_10_0_0 edge_3_11_0_0 edge_3_12_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0
 edge_4_11_0_0 edge_4_12_0_0 edge_6_1_0_0 edge_6_3_0_0 edge_6_4_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_6_12_0_0
 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0 edge_7_6_0_0
 edge_7_8_0_0 edge_7_10_0_0 edge_7_11_0_0 edge_7_12_0_0 edge_8_1_0_0
 edge_8_3_0_0 edge_8_4_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_10_0_0
 edge_8_11_0_0 edge_8_12_0_0 edge_10_1_0_0 edge_10_4_0_0 edge_10_8_0_0
 edge_10_11_0_0 edge_10_12_0_0 edge_11_1_0_0 edge_11_2_0_0 edge_11_3_0_0
 edge_11_4_0_0 edge_11_6_0_0 edge_11_7_0_0 edge_11_8_0_0 edge_11_10_0_0
 edge_11_12_0_0 edge_12_1_0_0 edge_12_11_0_0
End
