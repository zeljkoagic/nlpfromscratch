Maximize
  - 2.5239308458989 edge_0_2_0_0 - 1.0045958591282 edge_0_3_0_0
   - 5.107401696967201 edge_0_4_0_0 - 2.569153224962 edge_0_6_0_0
   - 1.4181218385021 edge_0_7_0_0 - 3.902608804828799 edge_0_9_0_0
   - 3.0299536111144 edge_0_10_0_0 - 2.8790573033013 edge_0_11_0_0
   + 0.6673751875247647 edge_2_3_0_0 + 0.3177783638371133 edge_2_4_0_0
   + 0.4106461446879793 edge_2_6_0_0 - 0.1913569997195838 edge_2_7_0_0
   - 0.9130766177005369 edge_2_8_0_0 + 0.0285949369465777 edge_2_9_0_0
   - 0.4645780875406934 edge_2_10_0_0 - 1.4541841311903 edge_2_11_0_0
   - 0.40644004437478 edge_2_12_0_0 + 2.952101163854695 edge_3_2_0_0
   + 0.7936818370504765 edge_3_3_0_0 + 2.370835510592831 edge_3_4_0_0
   + 1.765148596028195 edge_3_6_0_0 + 0.00726023183198589 edge_3_7_0_0
   - 0.0570958949485131 edge_3_8_0_0 + 0.8612932016601558 edge_3_9_0_0
   + 0.463988368072253 edge_3_10_0_0 + 0.9982514180057525 edge_3_11_0_0
   + 2.5886395340448 edge_3_12_0_0 - 0.6364281158841995 edge_4_2_0_0
   - 0.4830064196315688 edge_4_3_0_0 + 0.0775575543765099 edge_4_6_0_0
   - 0.3741504822334784 edge_4_7_0_0 - 0.6479083176740991 edge_4_8_0_0
   - 0.7450403670642138 edge_4_9_0_0 - 1.087260227827981 edge_4_10_0_0
   - 0.7162723049623201 edge_4_12_0_0 + 1.583741825881025 edge_6_2_0_0
   + 0.8940778840027019 edge_6_3_0_0 + 1.415304171801588 edge_6_4_0_0
   + 0.3948575767458243 edge_6_7_0_0 + 1.023553365086478 edge_6_8_0_0
   + 2.243677501867836 edge_6_9_0_0 + 0.9296205625870503 edge_6_10_0_0
   + 1.267706796669872 edge_6_11_0_0 + 2.39082821914 edge_6_12_0_0
   - 0.2484842211347586 edge_7_2_0_0 - 0.1532599852049624 edge_7_3_0_0
   + 0.0838845064014363 edge_7_4_0_0 + 0.1798473948334018 edge_7_6_0_0
   + 0.6104109760593157 edge_7_8_0_0 + 0.2257907904475812 edge_7_10_0_0
   - 0.9089894491716128 edge_8_3_0_0 - 0.2980323549250607 edge_8_4_0_0
   - 0.6491424132520766 edge_8_6_0_0 + 0.1078165808554219 edge_8_7_0_0
   + 0.7055205536061792 edge_8_9_0_0 + 0.5675245354740074 edge_8_10_0_0
   - 0.1101374093688427 edge_8_11_0_0 - 0.31450814186448 edge_8_12_0_0
   - 0.1527077982155475 edge_9_2_0_0 - 1.029462245698362 edge_9_3_0_0
   - 0.0819259284393446 edge_9_4_0_0 - 0.1977350929069166 edge_9_6_0_0
   + 1.572973214532896 edge_9_8_0_0 - 0.0993660042441839 edge_9_10_0_0
   + 0.2592994998592027 edge_9_11_0_0 - 0.077805551990928 edge_9_12_0_0
   - 0.3337787673191305 edge_10_2_0_0 - 0.0794704960726021 edge_10_3_0_0
   + 0.5703189717678491 edge_10_4_0_0 + 0.6397856960424493 edge_10_6_0_0
   + 0.5971369257871834 edge_10_7_0_0 + 2.782909918357759 edge_10_8_0_0
   - 0.616930496830087 edge_10_9_0_0 + 0.7868748564329399 edge_10_11_0_0
   + 0.22932034658838 edge_10_12_0_0 - 0.2829697136384196 edge_11_2_0_0
   + 0.5141506400381382 edge_11_3_0_0 - 0.5936181458270621 edge_11_4_0_0
   + 0.7326409617095604 edge_11_6_0_0 + 0.9180395366935491 edge_11_8_0_0
   + 1.41053496543017 edge_11_9_0_0 + 3.257622146094381 edge_11_10_0_0
   + 3.2126158323017 edge_11_12_0_0 - 0.58850340468854 edge_12_11_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_9_0_0
   - flow_2_10_0_0 - flow_2_11_0_0 - flow_2_12_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0
   + flow_10_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_3_3_0_0 + edge_4_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 + edge_9_3_0_0
   + edge_10_3_0_0 + edge_11_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 - flow_3_10_0_0 - flow_3_11_0_0 - flow_3_12_0_0
   + flow_4_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0
   + flow_9_3_0_0 + flow_10_3_0_0 + flow_11_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_6_4_0_0
   + edge_7_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0 + edge_10_4_0_0
   + edge_11_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_6_0_0 - flow_4_7_0_0
   - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_10_0_0 - flow_4_12_0_0
   + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0 + flow_9_4_0_0
   + flow_10_4_0_0 + flow_11_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0
   + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 - flow_6_12_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_6_7_0_0 + edge_8_7_0_0 + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0 - flow_7_3_0_0
   - flow_7_4_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_10_0_0
   + flow_8_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0 + edge_6_8_0_0
   + edge_7_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0 = 1
 consume_one_flow_unit_8: flow_2_8_0_0 + flow_3_8_0_0 + flow_4_8_0_0
   + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_3_0_0 - flow_8_4_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   - flow_8_11_0_0 - flow_8_12_0_0 + flow_9_8_0_0 + flow_10_8_0_0
   + flow_11_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_6_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 + edge_11_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_6_9_0_0 + flow_8_9_0_0 - flow_9_2_0_0
   - flow_9_3_0_0 - flow_9_4_0_0 - flow_9_6_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 - flow_9_11_0_0 - flow_9_12_0_0 + flow_10_9_0_0
   + flow_11_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_4_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0
   + edge_9_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_4_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0
   + flow_9_10_0_0 - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_6_0_0 - flow_10_7_0_0 - flow_10_8_0_0 - flow_10_9_0_0
   - flow_10_11_0_0 - flow_10_12_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_2_11_0_0 + edge_3_11_0_0
   + edge_6_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0
   + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_2_11_0_0 + flow_3_11_0_0
   + flow_6_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0
   - flow_11_2_0_0 - flow_11_3_0_0 - flow_11_4_0_0 - flow_11_6_0_0
   - flow_11_8_0_0 - flow_11_9_0_0 - flow_11_10_0_0 - flow_11_12_0_0
   + flow_12_11_0_0 = 1
 one_parent_12: edge_2_12_0_0 + edge_3_12_0_0 + edge_4_12_0_0
   + edge_6_12_0_0 + edge_8_12_0_0 + edge_9_12_0_0 + edge_10_12_0_0
   + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_2_12_0_0 + flow_3_12_0_0 + flow_4_12_0_0
   + flow_6_12_0_0 + flow_8_12_0_0 + flow_9_12_0_0 + flow_10_12_0_0
   + flow_11_12_0_0 - flow_12_11_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_9_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 12
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_3_0_0: flow_3_3_0_0 - 1000 edge_3_3_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_12_0_0: flow_4_12_0_0 - 1000 edge_4_12_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_3_0_0: flow_11_3_0_0 - 1000 edge_11_3_0_0 <= 0
 inactive_no_flow_11_4_0_0: flow_11_4_0_0 - 1000 edge_11_4_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0
 edge_2_11_0_0 edge_2_12_0_0 edge_3_2_0_0 edge_3_3_0_0 edge_3_4_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0
 edge_3_11_0_0 edge_3_12_0_0 edge_4_2_0_0 edge_4_3_0_0 edge_4_6_0_0
 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0 edge_4_10_0_0 edge_4_12_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_7_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_6_12_0_0 edge_7_2_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_10_0_0
 edge_8_3_0_0 edge_8_4_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_8_11_0_0 edge_8_12_0_0 edge_9_2_0_0 edge_9_3_0_0
 edge_9_4_0_0 edge_9_6_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_9_11_0_0
 edge_9_12_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0 edge_10_6_0_0
 edge_10_7_0_0 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_10_12_0_0
 edge_11_2_0_0 edge_11_3_0_0 edge_11_4_0_0 edge_11_6_0_0 edge_11_8_0_0
 edge_11_9_0_0 edge_11_10_0_0 edge_11_12_0_0 edge_12_11_0_0
End
