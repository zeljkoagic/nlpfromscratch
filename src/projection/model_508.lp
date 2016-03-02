Maximize
  - 3.5846889 edge_0_1_0_0 - 0.4356919263872 edge_0_8_0_0
   - 2.6413094912835 edge_0_9_0_0 - 2.4100502600412 edge_0_11_0_0
   - 0.8212206217503 edge_0_12_0_0 - 0.1184768947372231 edge_1_8_0_0
   - 0.6584540706904168 edge_1_9_0_0 - 0.5884512214320775 edge_1_10_0_0
   + 0.2031846953204627 edge_1_11_0_0 - 0.6102004220050067 edge_1_12_0_0
   - 0.7552095430057857 edge_1_14_0_0 - 0.8304489575465603 edge_1_19_0_0
   - 0.3503767783643897 edge_1_20_0_0 + 0.8772477861779687 edge_8_1_0_0
   + 1.213889609855568 edge_8_9_0_0 + 1.634108793683924 edge_8_10_0_0
   + 1.481611073824076 edge_8_11_0_0 + 1.249975187909707 edge_8_12_0_0
   + 0.4093613598383522 edge_8_14_0_0 + 0.5101513669300356 edge_8_19_0_0
   + 2.451842838433932 edge_8_20_0_0 - 0.0804201902250323 edge_9_1_0_0
   + 0.056020142826043 edge_9_8_0_0 + 1.683164666114323 edge_9_10_0_0
   - 0.0173239652200652 edge_9_11_0_0 + 0.6920640992226856 edge_9_12_0_0
   + 0.794024038388352 edge_9_14_0_0 + 0.1571493329998744 edge_9_19_0_0
   + 0.6970097060530595 edge_9_20_0_0 - 1.256253347549502 edge_10_1_0_0
   - 0.627738162105779 edge_10_8_0_0 - 0.1480213406997346 edge_10_9_0_0
   - 0.9459063503653304 edge_10_11_0_0 - 0.0200780308000994 edge_10_12_0_0
   - 0.3214482312605342 edge_11_1_0_0 + 0.4430209463506902 edge_11_8_0_0
   - 0.306380684947697 edge_11_9_0_0 + 0.0820334234487442 edge_11_10_0_0
   - 0.0590969290241244 edge_11_12_0_0 - 0.6821115745896021 edge_11_14_0_0
   - 0.5479296289356508 edge_11_19_0_0 - 0.0674132706417837 edge_11_20_0_0
   - 0.5275072963091776 edge_12_1_0_0 - 0.4433865667753532 edge_12_8_0_0
   - 0.05937138643235 edge_12_9_0_0 + 0.1559736988639647 edge_12_10_0_0
   - 0.4713394912343972 edge_12_11_0_0 + 0.4350389734299067 edge_12_14_0_0
   + 0.839046095438558 edge_12_19_0_0 + 1.250128132182972 edge_12_20_0_0
   - 0.9952068372249744 edge_14_1_0_0 - 0.6214055662503588 edge_14_8_0_0
   - 0.6861747651950997 edge_14_9_0_0 - 0.5596635878005943 edge_14_10_0_0
   - 0.3834318217699857 edge_14_12_0_0 + 0.1396219342229712 edge_14_19_0_0
   + 0.0860155947908525 edge_14_20_0_0 - 0.472610191709876 edge_19_1_0_0
   + 0.0202817324098097 edge_19_8_0_0 + 0.3585546102118272 edge_19_9_0_0
   - 0.115818244743802 edge_19_10_0_0 - 0.1168808928943029 edge_19_11_0_0
   + 0.1301039458634224 edge_19_12_0_0 + 1.276106123189848 edge_19_14_0_0
   + 0.8072596853404406 edge_19_20_0_0 - 0.8881887637220968 edge_20_14_0_0
   - 0.8543123492863324 edge_20_19_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_8_1_0_0 + edge_9_1_0_0 + edge_10_1_0_0
   + edge_11_1_0_0 + edge_12_1_0_0 + edge_14_1_0_0 + edge_19_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_8_0_0 - flow_1_9_0_0
   - flow_1_10_0_0 - flow_1_11_0_0 - flow_1_12_0_0 - flow_1_14_0_0
   - flow_1_19_0_0 - flow_1_20_0_0 + flow_8_1_0_0 + flow_9_1_0_0
   + flow_10_1_0_0 + flow_11_1_0_0 + flow_12_1_0_0 + flow_14_1_0_0
   + flow_19_1_0_0 = 1
 one_parent_2: = 1
 consume_one_flow_unit_2: = 1
 one_parent_3: = 1
 consume_one_flow_unit_3: = 1
 one_parent_4: = 1
 consume_one_flow_unit_4: = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: = 1
 consume_one_flow_unit_6: = 1
 one_parent_7: = 1
 consume_one_flow_unit_7: = 1
 one_parent_8: edge_0_8_0_0 + edge_1_8_0_0 + edge_9_8_0_0 + edge_10_8_0_0
   + edge_11_8_0_0 + edge_12_8_0_0 + edge_14_8_0_0 + edge_19_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_1_8_0_0 - flow_8_1_0_0
   - flow_8_9_0_0 - flow_8_10_0_0 - flow_8_11_0_0 - flow_8_12_0_0
   - flow_8_14_0_0 - flow_8_19_0_0 - flow_8_20_0_0 + flow_9_8_0_0
   + flow_10_8_0_0 + flow_11_8_0_0 + flow_12_8_0_0 + flow_14_8_0_0
   + flow_19_8_0_0 = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0
   + edge_11_9_0_0 + edge_12_9_0_0 + edge_14_9_0_0 + edge_19_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_8_9_0_0
   - flow_9_1_0_0 - flow_9_8_0_0 - flow_9_10_0_0 - flow_9_11_0_0
   - flow_9_12_0_0 - flow_9_14_0_0 - flow_9_19_0_0 - flow_9_20_0_0
   + flow_10_9_0_0 + flow_11_9_0_0 + flow_12_9_0_0 + flow_14_9_0_0
   + flow_19_9_0_0 = 1
 one_parent_10: edge_1_10_0_0 + edge_8_10_0_0 + edge_9_10_0_0
   + edge_11_10_0_0 + edge_12_10_0_0 + edge_14_10_0_0 + edge_19_10_0_0 = 1
 consume_one_flow_unit_10: flow_1_10_0_0 + flow_8_10_0_0 + flow_9_10_0_0
   - flow_10_1_0_0 - flow_10_8_0_0 - flow_10_9_0_0 - flow_10_11_0_0
   - flow_10_12_0_0 + flow_11_10_0_0 + flow_12_10_0_0 + flow_14_10_0_0
   + flow_19_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_8_11_0_0
   + edge_9_11_0_0 + edge_10_11_0_0 + edge_12_11_0_0 + edge_19_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_8_11_0_0
   + flow_9_11_0_0 + flow_10_11_0_0 - flow_11_1_0_0 - flow_11_8_0_0
   - flow_11_9_0_0 - flow_11_10_0_0 - flow_11_12_0_0 - flow_11_14_0_0
   - flow_11_19_0_0 - flow_11_20_0_0 + flow_12_11_0_0 + flow_19_11_0_0 = 1
 one_parent_12: edge_0_12_0_0 + edge_1_12_0_0 + edge_8_12_0_0
   + edge_9_12_0_0 + edge_10_12_0_0 + edge_11_12_0_0 + edge_14_12_0_0
   + edge_19_12_0_0 = 1
 consume_one_flow_unit_12: flow_0_12_0_0 + flow_1_12_0_0 + flow_8_12_0_0
   + flow_9_12_0_0 + flow_10_12_0_0 + flow_11_12_0_0 - flow_12_1_0_0
   - flow_12_8_0_0 - flow_12_9_0_0 - flow_12_10_0_0 - flow_12_11_0_0
   - flow_12_14_0_0 - flow_12_19_0_0 - flow_12_20_0_0 + flow_14_12_0_0
   + flow_19_12_0_0 = 1
 one_parent_13: = 1
 consume_one_flow_unit_13: = 1
 one_parent_14: edge_1_14_0_0 + edge_8_14_0_0 + edge_9_14_0_0
   + edge_11_14_0_0 + edge_12_14_0_0 + edge_19_14_0_0 + edge_20_14_0_0 = 1
 consume_one_flow_unit_14: flow_1_14_0_0 + flow_8_14_0_0 + flow_9_14_0_0
   + flow_11_14_0_0 + flow_12_14_0_0 - flow_14_1_0_0 - flow_14_8_0_0
   - flow_14_9_0_0 - flow_14_10_0_0 - flow_14_12_0_0 - flow_14_19_0_0
   - flow_14_20_0_0 + flow_19_14_0_0 + flow_20_14_0_0 = 1
 one_parent_15: = 1
 consume_one_flow_unit_15: = 1
 one_parent_16: = 1
 consume_one_flow_unit_16: = 1
 one_parent_17: = 1
 consume_one_flow_unit_17: = 1
 one_parent_18: = 1
 consume_one_flow_unit_18: = 1
 one_parent_19: edge_1_19_0_0 + edge_8_19_0_0 + edge_9_19_0_0
   + edge_11_19_0_0 + edge_12_19_0_0 + edge_14_19_0_0 + edge_20_19_0_0 = 1
 consume_one_flow_unit_19: flow_1_19_0_0 + flow_8_19_0_0 + flow_9_19_0_0
   + flow_11_19_0_0 + flow_12_19_0_0 + flow_14_19_0_0 - flow_19_1_0_0
   - flow_19_8_0_0 - flow_19_9_0_0 - flow_19_10_0_0 - flow_19_11_0_0
   - flow_19_12_0_0 - flow_19_14_0_0 - flow_19_20_0_0 + flow_20_19_0_0 = 1
 one_parent_20: edge_1_20_0_0 + edge_8_20_0_0 + edge_9_20_0_0
   + edge_11_20_0_0 + edge_12_20_0_0 + edge_14_20_0_0 + edge_19_20_0_0 = 1
 consume_one_flow_unit_20: flow_1_20_0_0 + flow_8_20_0_0 + flow_9_20_0_0
   + flow_11_20_0_0 + flow_12_20_0_0 + flow_14_20_0_0 + flow_19_20_0_0
   - flow_20_14_0_0 - flow_20_19_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_8_0_0 + flow_0_9_0_0 + flow_0_11_0_0
   + flow_0_12_0_0 = 20
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_0_12_0_0: flow_0_12_0_0 - 1000 edge_0_12_0_0 <= 0
 inactive_no_flow_1_8_0_0: flow_1_8_0_0 - 1000 edge_1_8_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_1_12_0_0: flow_1_12_0_0 - 1000 edge_1_12_0_0 <= 0
 inactive_no_flow_1_14_0_0: flow_1_14_0_0 - 1000 edge_1_14_0_0 <= 0
 inactive_no_flow_1_19_0_0: flow_1_19_0_0 - 1000 edge_1_19_0_0 <= 0
 inactive_no_flow_1_20_0_0: flow_1_20_0_0 - 1000 edge_1_20_0_0 <= 0
 inactive_no_flow_8_1_0_0: flow_8_1_0_0 - 1000 edge_8_1_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_8_12_0_0: flow_8_12_0_0 - 1000 edge_8_12_0_0 <= 0
 inactive_no_flow_8_14_0_0: flow_8_14_0_0 - 1000 edge_8_14_0_0 <= 0
 inactive_no_flow_8_19_0_0: flow_8_19_0_0 - 1000 edge_8_19_0_0 <= 0
 inactive_no_flow_8_20_0_0: flow_8_20_0_0 - 1000 edge_8_20_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_8_0_0: flow_9_8_0_0 - 1000 edge_9_8_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_9_12_0_0: flow_9_12_0_0 - 1000 edge_9_12_0_0 <= 0
 inactive_no_flow_9_14_0_0: flow_9_14_0_0 - 1000 edge_9_14_0_0 <= 0
 inactive_no_flow_9_19_0_0: flow_9_19_0_0 - 1000 edge_9_19_0_0 <= 0
 inactive_no_flow_9_20_0_0: flow_9_20_0_0 - 1000 edge_9_20_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_10_12_0_0: flow_10_12_0_0 - 1000 edge_10_12_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_8_0_0: flow_11_8_0_0 - 1000 edge_11_8_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
 inactive_no_flow_11_12_0_0: flow_11_12_0_0 - 1000 edge_11_12_0_0 <= 0
 inactive_no_flow_11_14_0_0: flow_11_14_0_0 - 1000 edge_11_14_0_0 <= 0
 inactive_no_flow_11_19_0_0: flow_11_19_0_0 - 1000 edge_11_19_0_0 <= 0
 inactive_no_flow_11_20_0_0: flow_11_20_0_0 - 1000 edge_11_20_0_0 <= 0
 inactive_no_flow_12_1_0_0: flow_12_1_0_0 - 1000 edge_12_1_0_0 <= 0
 inactive_no_flow_12_8_0_0: flow_12_8_0_0 - 1000 edge_12_8_0_0 <= 0
 inactive_no_flow_12_9_0_0: flow_12_9_0_0 - 1000 edge_12_9_0_0 <= 0
 inactive_no_flow_12_10_0_0: flow_12_10_0_0 - 1000 edge_12_10_0_0 <= 0
 inactive_no_flow_12_11_0_0: flow_12_11_0_0 - 1000 edge_12_11_0_0 <= 0
 inactive_no_flow_12_14_0_0: flow_12_14_0_0 - 1000 edge_12_14_0_0 <= 0
 inactive_no_flow_12_19_0_0: flow_12_19_0_0 - 1000 edge_12_19_0_0 <= 0
 inactive_no_flow_12_20_0_0: flow_12_20_0_0 - 1000 edge_12_20_0_0 <= 0
 inactive_no_flow_14_1_0_0: flow_14_1_0_0 - 1000 edge_14_1_0_0 <= 0
 inactive_no_flow_14_8_0_0: flow_14_8_0_0 - 1000 edge_14_8_0_0 <= 0
 inactive_no_flow_14_9_0_0: flow_14_9_0_0 - 1000 edge_14_9_0_0 <= 0
 inactive_no_flow_14_10_0_0: flow_14_10_0_0 - 1000 edge_14_10_0_0 <= 0
 inactive_no_flow_14_12_0_0: flow_14_12_0_0 - 1000 edge_14_12_0_0 <= 0
 inactive_no_flow_14_19_0_0: flow_14_19_0_0 - 1000 edge_14_19_0_0 <= 0
 inactive_no_flow_14_20_0_0: flow_14_20_0_0 - 1000 edge_14_20_0_0 <= 0
 inactive_no_flow_19_1_0_0: flow_19_1_0_0 - 1000 edge_19_1_0_0 <= 0
 inactive_no_flow_19_8_0_0: flow_19_8_0_0 - 1000 edge_19_8_0_0 <= 0
 inactive_no_flow_19_9_0_0: flow_19_9_0_0 - 1000 edge_19_9_0_0 <= 0
 inactive_no_flow_19_10_0_0: flow_19_10_0_0 - 1000 edge_19_10_0_0 <= 0
 inactive_no_flow_19_11_0_0: flow_19_11_0_0 - 1000 edge_19_11_0_0 <= 0
 inactive_no_flow_19_12_0_0: flow_19_12_0_0 - 1000 edge_19_12_0_0 <= 0
 inactive_no_flow_19_14_0_0: flow_19_14_0_0 - 1000 edge_19_14_0_0 <= 0
 inactive_no_flow_19_20_0_0: flow_19_20_0_0 - 1000 edge_19_20_0_0 <= 0
 inactive_no_flow_20_14_0_0: flow_20_14_0_0 - 1000 edge_20_14_0_0 <= 0
 inactive_no_flow_20_19_0_0: flow_20_19_0_0 - 1000 edge_20_19_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_8_0_0 edge_0_9_0_0 edge_0_11_0_0 edge_0_12_0_0
 edge_1_8_0_0 edge_1_9_0_0 edge_1_10_0_0 edge_1_11_0_0 edge_1_12_0_0
 edge_1_14_0_0 edge_1_19_0_0 edge_1_20_0_0 edge_8_1_0_0 edge_8_9_0_0
 edge_8_10_0_0 edge_8_11_0_0 edge_8_12_0_0 edge_8_14_0_0 edge_8_19_0_0
 edge_8_20_0_0 edge_9_1_0_0 edge_9_8_0_0 edge_9_10_0_0 edge_9_11_0_0
 edge_9_12_0_0 edge_9_14_0_0 edge_9_19_0_0 edge_9_20_0_0 edge_10_1_0_0
 edge_10_8_0_0 edge_10_9_0_0 edge_10_11_0_0 edge_10_12_0_0 edge_11_1_0_0
 edge_11_8_0_0 edge_11_9_0_0 edge_11_10_0_0 edge_11_12_0_0 edge_11_14_0_0
 edge_11_19_0_0 edge_11_20_0_0 edge_12_1_0_0 edge_12_8_0_0 edge_12_9_0_0
 edge_12_10_0_0 edge_12_11_0_0 edge_12_14_0_0 edge_12_19_0_0 edge_12_20_0_0
 edge_14_1_0_0 edge_14_8_0_0 edge_14_9_0_0 edge_14_10_0_0 edge_14_12_0_0
 edge_14_19_0_0 edge_14_20_0_0 edge_19_1_0_0 edge_19_8_0_0 edge_19_9_0_0
 edge_19_10_0_0 edge_19_11_0_0 edge_19_12_0_0 edge_19_14_0_0 edge_19_20_0_0
 edge_20_14_0_0 edge_20_19_0_0
End
