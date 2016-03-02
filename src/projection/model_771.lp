Maximize
  - 2.8495452107836 edge_0_1_0_0 - 3.1316113806098 edge_0_2_0_0
   - 4.0849310061846 edge_0_3_0_0 - 2.6228999841941 edge_0_4_0_0
   - 2.849098358002 edge_0_5_0_0 - 2.160494901642 edge_0_6_0_0
   - 3.0696327801045 edge_0_7_0_0 - 3.60980580532 edge_0_8_0_0
   - 4.5166065222103 edge_0_9_0_0 + 0.5762454612758341 edge_1_2_0_0
   + 1.109342478843017 edge_1_3_0_0 + 1.080762052001331 edge_1_4_0_0
   + 0.848325674160462 edge_1_5_0_0 + 0.5111498580293509 edge_1_6_0_0
   + 0.6049098768130292 edge_1_7_0_0 + 1.296859942494305 edge_1_8_0_0
   + 2.165047096251052 edge_1_9_0_0 + 1.315929708300968 edge_2_1_0_0
   + 0.8739609941656417 edge_2_4_0_0 + 0.4635030514028978 edge_2_6_0_0
   + 0.0711108498056275 edge_2_7_0_0 + 1.297746692355678 edge_2_9_0_0
   - 1.773766993113167 edge_3_1_0_0 - 0.3267294380318539 edge_3_4_0_0
   - 0.1380932406150872 edge_3_5_0_0 - 0.3902855174263262 edge_3_7_0_0
   - 0.4916340297966815 edge_3_8_0_0 - 0.3334067055455405 edge_3_9_0_0
   + 0.4382804404309601 edge_4_1_0_0 + 0.5016809376445674 edge_4_2_0_0
   + 0.5604594328112146 edge_4_3_0_0 + 1.887374876227948 edge_4_5_0_0
   + 0.9930037260531059 edge_4_6_0_0 + 1.959884041936808 edge_4_7_0_0
   + 0.3966666624036452 edge_4_8_0_0 + 1.933793560449812 edge_4_9_0_0
   + 0.9630533563951489 edge_5_1_0_0 - 0.225565064955729 edge_5_3_0_0
   + 2.050972658294554 edge_5_4_0_0 - 0.0117404807196679 edge_5_6_0_0
   + 2.011839572954541 edge_5_7_0_0 + 0.1513467719816944 edge_5_8_0_0
   + 1.765633171919411 edge_5_9_0_0 - 0.4160533168451186 edge_6_1_0_0
   - 0.3528444076921699 edge_6_2_0_0 - 0.4272861456617912 edge_6_3_0_0
   + 0.1469728640948486 edge_6_4_0_0 + 0.3546570652334237 edge_6_5_0_0
   + 0.5410535830340079 edge_6_7_0_0 + 0.0942320220215834 edge_6_8_0_0
   + 0.2912993474367898 edge_6_9_0_0 + 0.5785568246190709 edge_7_1_0_0
   - 0.7378480620953176 edge_7_2_0_0 + 0.9516670689413255 edge_7_3_0_0
   + 2.369346924528584 edge_7_4_0_0 + 1.902664333853681 edge_7_5_0_0
   + 3.628564375718363 edge_7_6_0_0 + 2.596229125027988 edge_7_8_0_0
   + 2.419577411049675 edge_7_9_0_0 - 0.3162568490922162 edge_8_1_0_0
   + 0.3492145098626013 edge_8_3_0_0 + 1.963466423298617 edge_8_4_0_0
   + 3.470356071633283 edge_8_5_0_0 + 1.429763781322962 edge_8_6_0_0
   + 1.076935149662636 edge_8_7_0_0 + 1.351080112150367 edge_8_9_0_0
   - 2.426055540632496 edge_9_1_0_0 - 1.744418018704491 edge_9_2_0_0
   - 1.611621894864486 edge_9_4_0_0 - 1.003607907567016 edge_9_6_0_0
   - 1.00238418886858 edge_9_7_0_0 - 0.8599884727445708 edge_9_8_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 + edge_8_1_0_0
   + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_7_0_0
   - flow_1_8_0_0 - flow_1_9_0_0 + flow_2_1_0_0 + flow_3_1_0_0
   + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0
   + flow_8_1_0_0 + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_4_2_0_0 + edge_6_2_0_0
   + edge_7_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_4_0_0 - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_9_0_0
   + flow_4_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 - flow_3_1_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_7_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_7_3_0_0 + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0
   + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0
   + flow_8_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_1_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0
   - flow_6_8_0_0 - flow_6_9_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0
   + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0
   + flow_8_7_0_0 + flow_9_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_1_0_0 - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0
   + edge_8_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0
   + flow_7_9_0_0 + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0
   - flow_9_4_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0
   + flow_0_9_0_0 = 9
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
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
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
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
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_7_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_4_0_0 edge_1_5_0_0 edge_1_6_0_0 edge_1_7_0_0
 edge_1_8_0_0 edge_1_9_0_0 edge_2_1_0_0 edge_2_4_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_9_0_0 edge_3_1_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_4_1_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0
 edge_4_9_0_0 edge_5_1_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_6_1_0_0 edge_6_2_0_0
 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0 edge_6_8_0_0
 edge_6_9_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_7_9_0_0 edge_8_1_0_0
 edge_8_3_0_0 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0
 edge_8_9_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_4_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_8_0_0
End
