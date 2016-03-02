Maximize
  - 2.6077796870664 edge_0_2_0_0 - 1.78695 edge_0_3_0_0
   - 2.6371200617212 edge_0_4_0_0 - 4.115195802432 edge_0_5_0_0
   - 4.5454346113748 edge_0_6_0_0 - 3.0298596459912 edge_0_7_0_0
   - 2.36593 edge_0_8_0_0 - 2.9356 edge_0_9_0_0
   - 4.424937610532401 edge_0_10_0_0 + 1.1306598643208 edge_2_3_0_0
   + 2.107121213269413 edge_2_4_0_0 + 1.352786523068798 edge_2_5_0_0
   + 2.16483094320086 edge_2_6_0_0 + 2.120331158183723 edge_2_7_0_0
   - 0.1213449854386 edge_2_8_0_0 + 0.91386883550358 edge_2_9_0_0
   + 1.22979918833208 edge_2_10_0_0 + 2.2474595954572 edge_3_2_0_0
   + 2.7289591114097 edge_3_4_0_0 + 1.792031129402 edge_3_5_0_0
   + 2.6531519269605 edge_3_6_0_0 + 0.45980716536209 edge_3_7_0_0
   + 1.18924 edge_3_8_0_0 + 0.26597 edge_3_9_0_0 + 2.74072 edge_3_10_0_0
   + 2.233850488644709 edge_4_2_0_0 + 3.1964252285548 edge_4_3_0_0
   + 1.684983306810805 edge_4_5_0_0 + 1.96981505264908 edge_4_6_0_0
   + 1.714896315045502 edge_4_7_0_0 + 0.22087714218284 edge_4_8_0_0
   + 1.5175497535132 edge_4_9_0_0 + 1.664954945123844 edge_4_10_0_0
   - 0.8895212889611525 edge_5_2_0_0 - 0.7422980276144799 edge_5_3_0_0
   - 0.1496846618101164 edge_5_4_0_0 + 0.535053127848185 edge_5_6_0_0
   + 0.5702008929780792 edge_5_7_0_0 + 0.4151437583646599 edge_5_8_0_0
   - 0.22684737078752 edge_5_9_0_0 + 0.2465708803023927 edge_5_10_0_0
   - 1.332677232454055 edge_6_2_0_0 - 1.198594191732 edge_6_3_0_0
   - 0.894754469765921 edge_6_4_0_0 + 0.6608076280563996 edge_6_5_0_0
   + 0.1647490766071615 edge_6_7_0_0 - 0.23429224313924 edge_6_8_0_0
   - 0.6694009840196 edge_6_9_0_0 - 0.27257903649025 edge_6_10_0_0
   - 0.7663088044795103 edge_7_2_0_0 - 0.40994636430748 edge_7_3_0_0
   - 0.3979288245089752 edge_7_4_0_0 + 0.1167309564182255 edge_7_5_0_0
   + 0.1859458636649937 edge_7_6_0_0 + 1.0784913976496 edge_7_8_0_0
   + 2.0431118479512 edge_7_9_0_0 + 0.8255907058799401 edge_7_10_0_0
   + 0.49488591092052 edge_8_2_0_0 + 0.270836 edge_8_3_0_0
   + 0.15540837991809 edge_8_4_0_0 + 0.55041627542595 edge_8_5_0_0
   + 0.4253786080593 edge_8_6_0_0 + 1.6815532905756 edge_8_7_0_0
   + 2.39819 edge_8_9_0_0 + 2.08352 edge_8_10_0_0
   + 0.71277091446748 edge_9_2_0_0 + 1.4271 edge_9_3_0_0
   + 0.49674082608392 edge_9_4_0_0 + 0.95042429537645 edge_9_5_0_0
   + 0.5165818069526401 edge_9_6_0_0 + 2.318738505424 edge_9_7_0_0
   + 3.40648 edge_9_8_0_0 + 1.92326 edge_9_10_0_0
   - 2.345028452280352 edge_10_2_0_0 - 2.812323896496275 edge_10_4_0_0
   - 1.831742767934971 edge_10_5_0_0 - 1.677766684590301 edge_10_6_0_0
   - 1.83784026672292 edge_10_7_0_0 - 1.38224 edge_10_8_0_0
   + 0.0842065 edge_10_9_0_0
Subject To
 one_parent_1: = 1
 consume_one_flow_unit_1: = 1
 one_parent_2: edge_0_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0 + edge_5_2_0_0
   + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0 + edge_9_2_0_0
   + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 - flow_2_3_0_0 - flow_2_4_0_0
   - flow_2_5_0_0 - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_8_0_0
   - flow_2_9_0_0 - flow_2_10_0_0 + flow_3_2_0_0 + flow_4_2_0_0
   + flow_5_2_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0
   + flow_9_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_6_3_0_0 + edge_7_3_0_0 + edge_8_3_0_0 + edge_9_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_7_0_0
   - flow_3_8_0_0 - flow_3_9_0_0 - flow_3_10_0_0 + flow_4_3_0_0
   + flow_5_3_0_0 + flow_6_3_0_0 + flow_7_3_0_0 + flow_8_3_0_0
   + flow_9_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_6_4_0_0 + edge_7_4_0_0 + edge_8_4_0_0 + edge_9_4_0_0
   + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_7_0_0 - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_10_0_0
   + flow_5_4_0_0 + flow_6_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0
   + flow_9_4_0_0 + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_6_5_0_0 + edge_7_5_0_0 + edge_8_5_0_0 + edge_9_5_0_0
   + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_7_0_0 - flow_5_8_0_0 - flow_5_9_0_0
   - flow_5_10_0_0 + flow_6_5_0_0 + flow_7_5_0_0 + flow_8_5_0_0
   + flow_9_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0 + edge_4_6_0_0
   + edge_5_6_0_0 + edge_7_6_0_0 + edge_8_6_0_0 + edge_9_6_0_0
   + edge_10_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_2_6_0_0 + flow_3_6_0_0
   + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_5_0_0 - flow_6_7_0_0 - flow_6_8_0_0
   - flow_6_9_0_0 - flow_6_10_0_0 + flow_7_6_0_0 + flow_8_6_0_0
   + flow_9_6_0_0 + flow_10_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_2_7_0_0 + edge_3_7_0_0 + edge_4_7_0_0
   + edge_5_7_0_0 + edge_6_7_0_0 + edge_8_7_0_0 + edge_9_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_2_7_0_0 + flow_3_7_0_0
   + flow_4_7_0_0 + flow_5_7_0_0 + flow_6_7_0_0 - flow_7_2_0_0
   - flow_7_3_0_0 - flow_7_4_0_0 - flow_7_5_0_0 - flow_7_6_0_0
   - flow_7_8_0_0 - flow_7_9_0_0 - flow_7_10_0_0 + flow_8_7_0_0
   + flow_9_7_0_0 + flow_10_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_5_8_0_0 + edge_6_8_0_0 + edge_7_8_0_0 + edge_9_8_0_0
   + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_5_8_0_0 + flow_6_8_0_0 + flow_7_8_0_0
   - flow_8_2_0_0 - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0
   - flow_8_6_0_0 - flow_8_7_0_0 - flow_8_9_0_0 - flow_8_10_0_0
   + flow_9_8_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_6_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0
   + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0
   + flow_8_9_0_0 - flow_9_2_0_0 - flow_9_3_0_0 - flow_9_4_0_0
   - flow_9_5_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_8_0_0
   - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_2_10_0_0 + edge_3_10_0_0
   + edge_4_10_0_0 + edge_5_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_2_10_0_0 + flow_3_10_0_0
   + flow_4_10_0_0 + flow_5_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_2_0_0 - flow_10_4_0_0
   - flow_10_5_0_0 - flow_10_6_0_0 - flow_10_7_0_0 - flow_10_8_0_0
   - flow_10_9_0_0 = 1
 root_flow: flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0 + flow_0_5_0_0
   + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 + flow_0_9_0_0
   + flow_0_10_0_0 = 10
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_2_3_0_0: flow_2_3_0_0 - 1000 edge_2_3_0_0 <= 0
 inactive_no_flow_2_4_0_0: flow_2_4_0_0 - 1000 edge_2_4_0_0 <= 0
 inactive_no_flow_2_5_0_0: flow_2_5_0_0 - 1000 edge_2_5_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_8_0_0: flow_2_8_0_0 - 1000 edge_2_8_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_7_0_0: flow_3_7_0_0 - 1000 edge_3_7_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_10_0_0: flow_3_10_0_0 - 1000 edge_3_10_0_0 <= 0
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
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_5_0_0: flow_6_5_0_0 - 1000 edge_6_5_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_8_0_0: flow_6_8_0_0 - 1000 edge_6_8_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
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
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_3_0_0: flow_9_3_0_0 - 1000 edge_9_3_0_0 <= 0
 inactive_no_flow_9_4_0_0: flow_9_4_0_0 - 1000 edge_9_4_0_0 <= 0
 inactive_no_flow_9_5_0_0: flow_9_5_0_0 - 1000 edge_9_5_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
Bounds
Binaries
 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0 edge_0_6_0_0
 edge_0_7_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_0_10_0_0 edge_2_3_0_0
 edge_2_4_0_0 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0
 edge_2_9_0_0 edge_2_10_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_7_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_10_0_0
 edge_4_2_0_0 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_7_0_0
 edge_4_8_0_0 edge_4_9_0_0 edge_4_10_0_0 edge_5_2_0_0 edge_5_3_0_0
 edge_5_4_0_0 edge_5_6_0_0 edge_5_7_0_0 edge_5_8_0_0 edge_5_9_0_0
 edge_5_10_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0 edge_6_5_0_0
 edge_6_7_0_0 edge_6_8_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_7_2_0_0
 edge_7_3_0_0 edge_7_4_0_0 edge_7_5_0_0 edge_7_6_0_0 edge_7_8_0_0
 edge_7_9_0_0 edge_7_10_0_0 edge_8_2_0_0 edge_8_3_0_0 edge_8_4_0_0
 edge_8_5_0_0 edge_8_6_0_0 edge_8_7_0_0 edge_8_9_0_0 edge_8_10_0_0
 edge_9_2_0_0 edge_9_3_0_0 edge_9_4_0_0 edge_9_5_0_0 edge_9_6_0_0
 edge_9_7_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_10_2_0_0 edge_10_4_0_0
 edge_10_5_0_0 edge_10_6_0_0 edge_10_7_0_0 edge_10_8_0_0 edge_10_9_0_0
End
