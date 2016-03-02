Maximize
  - 3.950723362595 edge_0_1_0_0 - 2.7312041601684 edge_0_2_0_0
   - 0.5939139494001999 edge_0_3_0_0 - 2.70821 edge_0_5_0_0
   - 3.5789056287075 edge_0_7_0_0 - 0.0372962186910738 edge_1_2_0_0
   - 0.141860170093573 edge_1_3_0_0 - 0.8083893862635401 edge_1_5_0_0
   - 1.294749199934757 edge_1_7_0_0 + 0.9538149411711034 edge_1_8_0_0
   - 0.0810484752379385 edge_2_1_0_0 + 0.5311670601131363 edge_2_3_0_0
   - 0.2064551795698673 edge_2_4_0_0 + 0.09701517243572 edge_2_5_0_0
   - 0.9525021346848216 edge_2_6_0_0 - 0.5135635372872976 edge_2_7_0_0
   + 0.4725446882827799 edge_2_8_0_0 + 0.829051578590618 edge_3_1_0_0
   + 1.820316888622652 edge_3_2_0_0 + 0.2345392996043694 edge_3_4_0_0
   + 3.793666339573736 edge_3_5_0_0 + 0.3888892835420215 edge_3_6_0_0
   + 2.497199011484222 edge_3_7_0_0 + 3.498652273739975 edge_3_8_0_0
   - 1.173844667176479 edge_4_1_0_0 - 0.1629795834620452 edge_4_2_0_0
   - 0.6595072192234188 edge_4_6_0_0 - 0.1923261553458378 edge_4_7_0_0
   - 0.2145065042163223 edge_5_1_0_0 + 0.4577238635423697 edge_5_2_0_0
   + 1.118288902166961 edge_5_3_0_0 + 3.7594024792128 edge_5_4_0_0
   + 0.7941997617577672 edge_5_6_0_0 + 3.187280740499363 edge_5_7_0_0
   + 1.833049670051 edge_5_8_0_0 - 0.9669881384935369 edge_6_1_0_0
   - 0.6029589454839971 edge_6_2_0_0 - 0.4039181465749367 edge_6_3_0_0
   - 0.54863135075432 edge_6_5_0_0 + 0.3901390328975732 edge_6_7_0_0
   + 0.00612985561559822 edge_6_8_0_0 - 0.2912988382852558 edge_7_1_0_0
   - 0.1983782046907446 edge_7_2_0_0 + 0.2357207913072446 edge_7_3_0_0
   + 0.9361476626584065 edge_7_4_0_0 + 0.4521498492270125 edge_7_5_0_0
   + 3.556148189080009 edge_7_6_0_0 + 1.269535044201318 edge_7_8_0_0
   - 0.8026055787484684 edge_8_7_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_3_1_0_0 + edge_4_1_0_0
   + edge_5_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_5_0_0 - flow_1_7_0_0 - flow_1_8_0_0 + flow_2_1_0_0
   + flow_3_1_0_0 + flow_4_1_0_0 + flow_5_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_8_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_1_3_0_0 + edge_2_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_1_3_0_0 + flow_2_3_0_0
   - flow_3_1_0_0 - flow_3_2_0_0 - flow_3_4_0_0 - flow_3_5_0_0
   - flow_3_6_0_0 - flow_3_7_0_0 - flow_3_8_0_0 + flow_5_3_0_0
   + flow_6_3_0_0 + flow_7_3_0_0 = 1
 one_parent_4: edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0 + edge_7_4_0_0
   = 1
 consume_one_flow_unit_4: flow_2_4_0_0 + flow_3_4_0_0 - flow_4_1_0_0
   - flow_4_2_0_0 - flow_4_6_0_0 - flow_4_7_0_0 + flow_5_4_0_0
   + flow_7_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_1_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_1_5_0_0 + flow_2_5_0_0
   + flow_3_5_0_0 - flow_5_1_0_0 - flow_5_2_0_0 - flow_5_3_0_0
   - flow_5_4_0_0 - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0
   + flow_6_5_0_0 + flow_7_5_0_0 = 1
 one_parent_6: edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0 + edge_5_6_0_0
   + edge_7_6_0_0 = 1
 consume_one_flow_unit_6: flow_2_6_0_0 + flow_3_6_0_0 + flow_4_6_0_0
   + flow_5_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0 + flow_7_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0
   + edge_4_7_0_0 + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_3_7_0_0 + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_3_0_0 - flow_7_4_0_0
   - flow_7_5_0_0 - flow_7_6_0_0 - flow_7_8_0_0 + flow_8_7_0_0 = 1
 one_parent_8: edge_1_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_5_8_0_0
   + edge_6_8_0_0 + edge_7_8_0_0 = 1
 consume_one_flow_unit_8: flow_1_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0 - flow_8_7_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_5_0_0
   + flow_0_7_0_0 = 8
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_5_0_0: flow_1_5_0_0 - 1000 edge_1_5_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_3_1_0_0: flow_3_1_0_0 - 1000 edge_3_1_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_4_1_0_0: flow_4_1_0_0 - 1000 edge_4_1_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_7_0_0: flow_4_7_0_0 - 1000 edge_4_7_0_0 <= 0
 inactive_no_flow_5_1_0_0: flow_5_1_0_0 - 1000 edge_5_1_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_7_0_0: flow_5_7_0_0 - 1000 edge_5_7_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_3_0_0: flow_7_3_0_0 - 1000 edge_7_3_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_5_0_0: flow_7_5_0_0 - 1000 edge_7_5_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_8_7_0_0: flow_8_7_0_0 - 1000 edge_8_7_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_5_0_0 edge_0_7_0_0
 edge_1_2_0_0 edge_1_3_0_0 edge_1_5_0_0 edge_1_7_0_0 edge_1_8_0_0
 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0
 edge_2_7_0_0 edge_2_8_0_0 edge_3_1_0_0 edge_3_2_0_0 edge_3_4_0_0
 edge_3_5_0_0 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_4_1_0_0
 edge_4_2_0_0 edge_4_6_0_0 edge_4_7_0_0 edge_5_1_0_0 edge_5_2_0_0
 edge_5_3_0_0 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0
 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_5_0_0 edge_6_7_0_0
 edge_6_8_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_3_0_0 edge_7_4_0_0
 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0 edge_8_7_0_0
End
