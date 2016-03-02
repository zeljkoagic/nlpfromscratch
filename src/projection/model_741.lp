Maximize
  - 3.757430406365 edge_0_1_0_0 - 2.5065198496088 edge_0_2_0_0
   - 0.42903857039253 edge_0_3_0_0 - 2.8645393125104 edge_0_6_0_0
   - 0.2637107587336036 edge_1_2_0_0 - 0.3314765630837641 edge_1_3_0_0
   - 1.09760897295838 edge_1_5_0_0 - 1.273630612671079 edge_1_6_0_0
   - 0.8934431516014961 edge_1_8_0_0 - 1.580900737627358 edge_1_9_0_0
   - 1.614160753119858 edge_1_10_0_0 - 1.778883086805 edge_1_11_0_0
   - 0.7468970156021436 edge_1_12_0_0 + 0.6354682816514008 edge_2_1_0_0
   + 0.9428246290080788 edge_2_3_0_0 - 0.0598764177296527 edge_2_5_0_0
   + 0.5853658243902086 edge_2_6_0_0 - 0.2535899436198025 edge_2_8_0_0
   + 0.0460383054349895 edge_2_9_0_0 - 0.7369788629381456 edge_2_10_0_0
   - 0.26007098439574 edge_2_11_0_0 + 0.6604996037000214 edge_2_12_0_0
   + 2.149578737557363 edge_3_1_0_0 + 1.95008338743919 edge_3_2_0_0
   + 1.070748658698894 edge_3_5_0_0 + 2.394200063389947 edge_3_6_0_0
   + 0.4346300087251487 edge_3_8_0_0 + 1.855106470522786 edge_3_9_0_0
   + 0.5580431093277854 edge_3_10_0_0 + 1.5603247960811 edge_3_11_0_0
   + 4.001751904878905 edge_3_12_0_0 - 0.8275835298377454 edge_5_1_0_0
   - 0.5827532229476746 edge_5_2_0_0 + 0.8281011802155291 edge_5_6_0_0
   + 0.1679943802648183 edge_5_8_0_0 - 0.1050579976587735 edge_5_9_0_0
   - 0.629007547146999 edge_5_10_0_0 - 0.8250047410129899 edge_5_11_0_0
   - 0.1208954429618855 edge_5_12_0_0 - 0.3294928401426994 edge_6_1_0_0
   - 0.4921108523667072 edge_6_2_0_0 + 0.4361622593312826 edge_6_3_0_0
   + 2.676718322248349 edge_6_5_0_0 + 0.5232800156927662 edge_6_8_0_0
   + 1.920300697778322 edge_6_9_0_0 + 0.416356337655045 edge_6_10_0_0
   + 1.4688596474736 edge_6_11_0_0 + 1.127339120674946 edge_6_12_0_0
   - 0.2128304327046618 edge_8_1_0_0 - 0.0366149988518039 edge_8_2_0_0
   + 0.2903821760943092 edge_8_5_0_0 + 0.507804407616833 edge_8_6_0_0
   + 1.340539952845973 edge_8_9_0_0 + 1.072672621088694 edge_8_10_0_0
   + 0.3851579729088 edge_8_11_0_0 + 0.7466425106816265 edge_8_12_0_0
   - 1.017333991826871 edge_9_1_0_0 - 1.336824616851118 edge_9_2_0_0
   + 0.885609347428249 edge_9_5_0_0 + 0.2347177283427317 edge_9_6_0_0
   + 1.500765977915049 edge_9_8_0_0 + 0.735890393302935 edge_9_10_0_0
   + 2.2181150162908 edge_9_11_0_0 + 1.089443852602402 edge_9_12_0_0
   - 0.322772048532506 edge_10_2_0_0 + 0.1091558997102282 edge_10_5_0_0
   - 0.0144081932464598 edge_10_6_0_0 + 0.3087927028238126 edge_10_8_0_0
   + 0.7261696330796937 edge_10_9_0_0 + 1.8912972390567 edge_10_11_0_0
   + 0.28685645615787 edge_10_12_0_0 - 0.996615519534 edge_11_1_0_0
   - 1.3277199203368 edge_11_2_0_0 + 0.41586156297188 edge_11_5_0_0
   - 0.46734788783648 edge_11_6_0_0 + 0.8826362782848 edge_11_8_0_0
   + 0.60707506139456 edge_11_9_0_0 + 2.6628872738934 edge_11_10_0_0
   + 0.7473885964099399 edge_11_12_0_0 - 2.307441817553246 edge_12_1_0_0
   - 1.067561058563117 edge_12_8_0_0 - 2.164691588009111 edge_12_9_0_0
   - 1.157938292831084 edge_12_10_0_0 - 1.513349182791 edge_12_11_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_5_1_0_0
   + edge_6_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0 + edge_11_1_0_0
   + edge_12_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_6_0_0 - flow_1_8_0_0 - flow_1_9_0_0
   - flow_1_10_0_0 - flow_1_11_0_0 - flow_1_12_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0 + flow_8_1_0_0
   + flow_9_1_0_0 + flow_11_1_0_0 + flow_12_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 + edge_10_2_0_0
   + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_8_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 - flow_2_11_0_0 - flow_2_12_0_0
   + flow_3_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_8_2_0_0
   + flow_9_2_0_0 + flow_10_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_6_3_0_0
   = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_5_0_0 - flow_3_6_0_0
   - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0 - flow_3_11_0_0
   - flow_3_12_0_0 + flow_6_3_0_0 = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_6_5_0_0
   + edge_8_5_0_0 + edge_9_5_0_0 + edge_10_5_0_0 + edge_11_5_0_0 = 1
 consume_one_flow_unit_5: flow_1_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_6_0_0 - flow_5_8_0_0
   - flow_5_9_0_0 - flow_5_10_0_0 - flow_5_11_0_0 - flow_5_12_0_0
   + flow_6_5_0_0 + flow_8_5_0_0 + flow_9_5_0_0 + flow_10_5_0_0
   + flow_11_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_5_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0
   + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0
   - flow_6_3_0_0 - flow_6_5_0_0 - flow_6_8_0_0 - flow_6_9_0_0
   - flow_6_10_0_0 - flow_6_11_0_0 - flow_6_12_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 + flow_11_6_0_0 = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0 + edge_11_8_0_0
   + edge_12_8_0_0 = 1
 consume_one_flow_unit_8: flow_1_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 - flow_8_1_0_0 - flow_8_2_0_0
   - flow_8_5_0_0 - flow_8_6_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   - flow_8_11_0_0 - flow_8_12_0_0 + flow_9_8_0_0 + flow_10_8_0_0
   + flow_11_8_0_0 + flow_12_8_0_0 = 1
 one_parent_9: edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_5_9_0_0
   + edge_6_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 + edge_11_9_0_0
   + edge_12_9_0_0 = 1
 consume_one_flow_unit_9: flow_1_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_5_9_0_0 + flow_6_9_0_0 + flow_8_9_0_0 - flow_9_1_0_0
   - flow_9_2_0_0 - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 - flow_9_11_0_0 - flow_9_12_0_0 + flow_10_9_0_0
   + flow_11_9_0_0 + flow_12_9_0_0 = 1
 one_parent_10: edge_1_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_5_10_0_0 + edge_6_10_0_0 + edge_8_10_0_0 + edge_9_10_0_0
   + edge_11_10_0_0 + edge_12_10_0_0 = 1
 consume_one_flow_unit_10: flow_1_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_5_10_0_0 + flow_6_10_0_0 + flow_8_10_0_0 + flow_9_10_0_0
   - flow_10_2_0_0 - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_8_0_0
   - flow_10_9_0_0 - flow_10_11_0_0 - flow_10_12_0_0 + flow_11_10_0_0
   + flow_12_10_0_0 = 1
 one_parent_11: edge_1_11_0_0 + edge_2_11_0_0 + edge_3_11_0_0
   + edge_5_11_0_0 + edge_6_11_0_0 + edge_8_11_0_0 + edge_9_11_0_0
   + edge_10_11_0_0 + edge_12_11_0_0 = 1
 consume_one_flow_unit_11: flow_1_11_0_0 + flow_2_11_0_0 + flow_3_11_0_0
   + flow_5_11_0_0 + flow_6_11_0_0 + flow_8_11_0_0 + flow_9_11_0_0
   + flow_10_11_0_0 - flow_11_1_0_0 - flow_11_2_0_0 - flow_11_5_0_0
   - flow_11_6_0_0 - flow_11_8_0_0 - flow_11_9_0_0 - flow_11_10_0_0
   - flow_11_12_0_0 + flow_12_11_0_0 = 1
 one_parent_12: edge_1_12_0_0 + edge_2_12_0_0 + edge_3_12_0_0
   + edge_5_12_0_0 + edge_6_12_0_0 + edge_8_12_0_0 + edge_9_12_0_0
   + edge_10_12_0_0 + edge_11_12_0_0 = 1
 consume_one_flow_unit_12: flow_1_12_0_0 + flow_2_12_0_0 + flow_3_12_0_0
   + flow_5_12_0_0 + flow_6_12_0_0 + flow_8_12_0_0 + flow_9_12_0_0
   + flow_10_12_0_0 + flow_11_12_0_0 - flow_12_1_0_0 - flow_12_8_0_0
   - flow_12_9_0_0 - flow_12_10_0_0 - flow_12_11_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_6_0_0 = 12
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_2_12_0_0: flow_2_12_0_0 - 1000 edge_2_12_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_3_12_0_0: flow_3_12_0_0 - 1000 edge_3_12_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_5_12_0_0: flow_5_12_0_0 - 1000 edge_5_12_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_6_12_0_0: flow_6_12_0_0 - 1000 edge_6_12_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_5_0_0: flow_11_5_0_0 - 1000 edge_11_5_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_6_0_0 edge_1_2_0_0
 edge_1_3_0_0 edge_1_5_0_0 edge_1_6_0_0 edge_1_8_0_0 edge_1_9_0_0
 edge_1_10_0_0 edge_1_11_0_0 edge_1_12_0_0 edge_2_1_0_0 edge_2_3_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0
 edge_2_11_0_0 edge_2_12_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0 edge_3_11_0_0
 edge_3_12_0_0 edge_5_1_0_0 edge_5_2_0_0 edge_5_6_0_0 edge_5_8_0_0
 edge_5_9_0_0 edge_5_10_0_0 edge_5_11_0_0 edge_5_12_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_5_0_0 edge_6_8_0_0 edge_6_9_0_0
 edge_6_10_0_0 edge_6_11_0_0 edge_6_12_0_0 edge_8_1_0_0 edge_8_2_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_9_0_0 edge_8_10_0_0 edge_8_11_0_0
 edge_8_12_0_0 edge_9_1_0_0 edge_9_2_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_8_0_0 edge_9_10_0_0 edge_9_11_0_0 edge_9_12_0_0 edge_10_2_0_0
 edge_10_5_0_0 edge_10_6_0_0 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0
 edge_10_12_0_0 edge_11_1_0_0 edge_11_2_0_0 edge_11_5_0_0 edge_11_6_0_0
 edge_11_8_0_0 edge_11_9_0_0 edge_11_10_0_0 edge_11_12_0_0 edge_12_1_0_0
 edge_12_8_0_0 edge_12_9_0_0 edge_12_10_0_0 edge_12_11_0_0
End
