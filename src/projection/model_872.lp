Maximize
  - 3.3380008284792 edge_0_1_0_0 - 2.1174274716435 edge_0_2_0_0
   - 2.1757111835472 edge_0_3_0_0 - 3.0838338524577 edge_0_4_0_0
   - 1.0721034417108 edge_0_5_0_0 - 1.0828377368644 edge_0_6_0_0
   - 4.067406094926699 edge_0_7_0_0 - 3.4870467308413 edge_0_8_0_0
   - 3.0977651560522 edge_0_9_0_0 - 0.5941004778160114 edge_1_3_0_0
   - 0.2092587910830999 edge_1_4_0_0 - 0.0420778985131723 edge_1_6_0_0
   - 0.6108123964410259 edge_1_7_0_0 - 0.6870093100601994 edge_1_9_0_0
   - 0.3830937731971538 edge_2_1_0_0 + 0.9128213736818476 edge_2_3_0_0
   + 0.5595794965922953 edge_2_4_0_0 + 0.8449848859872509 edge_2_5_0_0
   + 0.1720215664645747 edge_2_6_0_0 + 0.3594535092855063 edge_2_7_0_0
   + 0.8641531364325723 edge_2_8_0_0 + 0.1159401075354229 edge_2_9_0_0
   + 0.6101566419154378 edge_2_10_0_0 - 0.0424411896913756 edge_3_1_0_0
   + 3.416498029131534 edge_3_2_0_0 + 2.032834692549798 edge_3_4_0_0
   + 2.454251611231455 edge_3_5_0_0 + 1.574272391106151 edge_3_6_0_0
   + 1.012060395606017 edge_3_7_0_0 + 0.4866499856121704 edge_3_8_0_0
   + 0.8605665238702359 edge_3_9_0_0 + 2.698885251907686 edge_3_10_0_0
   - 0.9337327994379551 edge_4_1_0_0 - 0.3486279366284467 edge_4_2_0_0
   - 0.131378112881963 edge_4_3_0_0 - 0.0451786900273326 edge_4_5_0_0
   - 0.468128696302231 edge_4_6_0_0 - 0.359958179057523 edge_4_7_0_0
   - 0.448701966675821 edge_4_8_0_0 - 0.4172581155327195 edge_4_9_0_0
   - 0.3553609490396591 edge_4_10_0_0 + 0.0243649698293304 edge_5_2_0_0
   - 0.2155993546294078 edge_5_3_0_0 + 0.1204845774995178 edge_5_4_0_0
   + 0.4335886217436117 edge_5_6_0_0 - 0.17391344799767 edge_5_7_0_0
   - 0.5283529455918625 edge_5_8_0_0 + 0.2878518393729098 edge_5_9_0_0
   - 0.0499509320941763 edge_5_10_0_0 + 1.605183251741992 edge_6_1_0_0
   + 1.029142196110964 edge_6_2_0_0 + 2.815246410485905 edge_6_3_0_0
   + 1.779836411072226 edge_6_4_0_0 + 0.6242533242531783 edge_6_5_0_0
   + 2.393175754384293 edge_6_7_0_0 + 1.576082968224948 edge_6_8_0_0
   + 2.464429274008052 edge_6_9_0_0 + 4.627587766455755 edge_6_10_0_0
   + 0.0669467817115918 edge_7_1_0_0 - 0.1733164894700611 edge_7_2_0_0
   + 0.341551970475334 edge_7_3_0_0 + 0.5194501114224275 edge_7_4_0_0
   - 0.0537857192420717 edge_7_5_0_0 + 0.2146778349828791 edge_7_6_0_0
   + 0.8896306789902048 edge_7_8_0_0 + 1.135312849938441 edge_7_9_0_0
   + 0.6429177366921299 edge_7_10_0_0 + 0.1030179172858906 edge_8_2_0_0
   - 0.1585759356211424 edge_8_3_0_0 + 0.18932122565448 edge_8_4_0_0
   + 0.2079100246271717 edge_8_5_0_0 + 0.9091207060944166 edge_8_6_0_0
   + 0.9136555452087032 edge_8_7_0_0 - 0.2270526313074882 edge_8_9_0_0
   - 0.1044573932774692 edge_8_10_0_0 + 0.2356075675802821 edge_9_1_0_0
   + 0.53464527444336 edge_9_2_0_0 + 0.8828333730663795 edge_9_3_0_0
   + 0.4871486740103627 edge_9_4_0_0 + 0.9864425801772714 edge_9_5_0_0
   + 1.0074799395512 edge_9_6_0_0 + 1.2104805143093 edge_9_7_0_0
   + 3.6578343642456 edge_9_8_0_0 + 1.530806321671703 edge_9_10_0_0
   - 1.170448552291726 edge_10_3_0_0 - 1.003943090428814 edge_10_6_0_0
   - 1.220086604761673 edge_10_9_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_6_1_0_0 + edge_7_1_0_0 + edge_9_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_3_0_0 - flow_1_4_0_0
   - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_9_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_6_1_0_0 + flow_7_1_0_0
   + flow_9_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_1_0_0 - flow_2_3_0_0
   - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0
   - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_10_0_0 + flow_3_2_0_0
   + flow_4_2_0_0 + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0
   + flow_8_2_0_0 + flow_9_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0
   + edge_5_3_0_0 + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0
   + edge_9_3_0_0 + edge_10_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 - flow_3_9_0_0
   - flow_3_10_0_0 + flow_4_3_0_0 + flow_5_3_0_0 + flow_6_3_0_0
   + flow_7_3_0_0 + flow_8_3_0_0 + flow_9_3_0_0 + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0
   + edge_5_4_0_0 + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0
   + edge_9_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 + flow_2_4_0_0
   + flow_3_4_0_0 - flow_4_1_0_0 - flow_4_2_0_0 - flow_4_3_0_0
   - flow_4_5_0_0 - flow_4_6_0_0 - flow_4_7_0_0 - flow_4_8_0_0
   - flow_4_9_0_0 - flow_4_10_0_0 + flow_5_4_0_0 + flow_6_4_0_0
   + flow_7_4_0_0 + flow_8_4_0_0 + flow_9_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0
   - flow_5_10_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0
   + flow_9_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0
   + edge_9_6_0_0 + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_1_0_0
   - flow_6_2_0_0 - flow_6_3_0_0 - flow_6_4_0_0 - flow_6_5_0_0
   - flow_6_7_0_0 - flow_6_8_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   + flow_7_6_0_0 + flow_8_6_0_0 + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0
   + edge_9_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 - flow_7_9_0_0
   - flow_7_10_0_0 + flow_8_7_0_0 + flow_9_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   + flow_9_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0
   + edge_4_9_0_0 + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0
   + edge_8_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_3_9_0_0 + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0
   + flow_7_9_0_0 + flow_8_9_0_0 - flow_9_1_0_0 - flow_9_2_0_0
   - flow_9_3_0_0 - flow_9_4_0_0 - flow_9_5_0_0 - flow_9_6_0_0
   - flow_9_7_0_0 - flow_9_8_0_0 - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_2_10_0_0 + edge_3_10_0_0 + edge_4_10_0_0
   + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_8_10_0_0
   + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_3_10_0_0 + flow_4_10_0_0
   + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_8_10_0_0
   + flow_9_10_0_0 - flow_10_3_0_0 - flow_10_6_0_0 - flow_10_9_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0
   + flow_0_9_0_0 = 10
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_10_0_0: flow_5_10_0_0 - 1000 edge_5_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_7_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_9_0_0 edge_2_1_0_0
 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0
 edge_2_8_0_0 edge_2_9_0_0 edge_2_10_0_0 edge_3_1_0_0 edge_3_2_0_0
 edge_3_4_0_0 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0
 edge_3_9_0_0 edge_3_10_0_0 edge_4_1_0_0 edge_4_2_0_0 edge_4_3_0_0
 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_4_8_0_0 edge_4_9_0_0
 edge_4_10_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0
 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_10_0_0 edge_6_1_0_0
 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_7_1_0_0 edge_7_2_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0
 edge_7_9_0_0 edge_7_10_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_8_10_0_0
 edge_9_1_0_0 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0
 edge_9_6_0_0 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_10_3_0_0
 edge_10_6_0_0 edge_10_9_0_0
End
