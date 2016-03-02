Maximize
  - 2.237160292164 edge_0_1_0_0 - 1.5205277509624 edge_0_2_0_0
   - 1.9704643947124 edge_0_3_0_0 - 2.2102611377616 edge_0_4_0_0
   - 2.69996104791 edge_0_5_0_0 - 2.2364317202353 edge_0_6_0_0
   - 1.3825477940734 edge_0_7_0_0 - 2.4123930901182 edge_0_8_0_0
   + 0.1643035181197117 edge_1_2_0_0 - 0.0803552555029618 edge_1_6_0_0
   - 0.2754659426652898 edge_1_7_0_0 - 0.0734227616327663 edge_1_9_0_0
   + 0.3332283082293139 edge_2_1_0_0 + 0.2496568715197161 edge_2_3_0_0
   - 0.6850257351583378 edge_2_4_0_0 - 0.2882148320625244 edge_2_5_0_0
   + 0.3432131526309841 edge_2_6_0_0 + 1.743672734090758 edge_2_7_0_0
   + 0.0540937271980986 edge_2_8_0_0 + 0.4024755971839947 edge_2_9_0_0
   - 0.420656 edge_2_10_0_0 + 0.2029442646457714 edge_2_11_0_0
   + 1.459106146410425 edge_3_2_0_0 - 0.4034841427051485 edge_3_4_0_0
   + 0.4018454933360252 edge_3_5_0_0 + 0.9415658377438584 edge_3_6_0_0
   + 0.5282839214914389 edge_3_8_0_0 + 0.256032882611391 edge_3_9_0_0
   - 0.0781202564180443 edge_3_11_0_0 - 0.5521191419810078 edge_4_2_0_0
   - 0.0496610675272088 edge_4_3_0_0 + 1.319844390177183 edge_4_5_0_0
   + 0.7904209579654904 edge_4_6_0_0 + 0.4931275931058828 edge_4_8_0_0
   + 0.0505990914431131 edge_4_9_0_0 - 0.0658505529409665 edge_4_10_0_0
   + 0.9085616652750969 edge_4_11_0_0 - 0.2272169312829002 edge_5_2_0_0
   + 0.5265865126869729 edge_5_3_0_0 + 3.937158420329969 edge_5_4_0_0
   + 0.720195506139806 edge_5_5_0_0 + 1.611119054240762 edge_5_6_0_0
   + 0.4412492992020193 edge_5_8_0_0 + 0.8643816133897483 edge_5_9_0_0
   + 0.1134836360182614 edge_5_11_0_0 - 0.1533603195276028 edge_6_2_0_0
   - 0.1722617592735144 edge_7_1_0_0 + 0.8200829807310382 edge_7_2_0_0
   - 0.7437231236813561 edge_7_4_0_0 - 0.0887953894420138 edge_7_8_0_0
   + 0.0843352942705234 edge_7_9_0_0 - 0.3432564484175699 edge_7_10_0_0
   - 0.1419802862898923 edge_7_11_0_0 - 0.3666565715617855 edge_8_2_0_0
   - 0.2535800290857047 edge_8_3_0_0 - 0.1895197817116524 edge_8_4_0_0
   - 0.1411380724689199 edge_8_5_0_0 + 0.5027826854465341 edge_8_6_0_0
   + 1.784249703779202 edge_8_9_0_0 + 1.026381063548612 edge_8_10_0_0
   + 1.066480633090158 edge_8_11_0_0 - 0.1008819195445503 edge_9_7_0_0
   - 0.4197658001768835 edge_9_10_0_0
   + 8.3714855150633129e-04 edge_10_1_0_0
   + 0.3217158971080794 edge_10_2_0_0 - 0.2359127940519114 edge_10_4_0_0
   - 0.1166679044030246 edge_10_5_0_0 + 0.1834880925477817 edge_10_7_0_0
   - 0.4546618725831376 edge_10_8_0_0 + 0.3528212902503125 edge_10_9_0_0
   - 0.1283916721115191 edge_10_10_0_0 + 0.764573825391187 edge_10_11_0_0
   - 0.3129521778856595 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_7_1_0_0 + edge_10_1_0_0
   = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_6_0_0
   - flow_1_7_0_0 - flow_1_9_0_0 + flow_2_1_0_0 + flow_7_1_0_0
   + flow_10_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_3_2_0_0 + edge_4_2_0_0
   + edge_5_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0 + edge_8_2_0_0
   + edge_10_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_3_0_0 - flow_2_4_0_0 - flow_2_5_0_0 - flow_2_6_0_0
   - flow_2_7_0_0 - flow_2_8_0_0 - flow_2_9_0_0 - flow_2_10_0_0
   - flow_2_11_0_0 + flow_3_2_0_0 + flow_4_2_0_0 + flow_5_2_0_0
   + flow_6_2_0_0 + flow_7_2_0_0 + flow_8_2_0_0 + flow_10_2_0_0 = 1
 one_parent_3: edge_0_3_0_0 + edge_2_3_0_0 + edge_4_3_0_0 + edge_5_3_0_0
   + edge_8_3_0_0 = 1
 consume_one_flow_unit_3: flow_0_3_0_0 + flow_2_3_0_0 - flow_3_2_0_0
   - flow_3_4_0_0 - flow_3_5_0_0 - flow_3_6_0_0 - flow_3_8_0_0
   - flow_3_9_0_0 - flow_3_11_0_0 + flow_4_3_0_0 + flow_5_3_0_0
   + flow_8_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_2_4_0_0 + edge_3_4_0_0 + edge_5_4_0_0
   + edge_7_4_0_0 + edge_8_4_0_0 + edge_10_4_0_0 = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_2_4_0_0 + flow_3_4_0_0
   - flow_4_2_0_0 - flow_4_3_0_0 - flow_4_5_0_0 - flow_4_6_0_0
   - flow_4_8_0_0 - flow_4_9_0_0 - flow_4_10_0_0 - flow_4_11_0_0
   + flow_5_4_0_0 + flow_7_4_0_0 + flow_8_4_0_0 + flow_10_4_0_0 = 1
 one_parent_5: edge_0_5_0_0 + edge_2_5_0_0 + edge_3_5_0_0 + edge_4_5_0_0
   + edge_5_5_0_0 + edge_8_5_0_0 + edge_10_5_0_0 = 1
 consume_one_flow_unit_5: flow_0_5_0_0 + flow_2_5_0_0 + flow_3_5_0_0
   + flow_4_5_0_0 - flow_5_2_0_0 - flow_5_3_0_0 - flow_5_4_0_0
   - flow_5_6_0_0 - flow_5_8_0_0 - flow_5_9_0_0 - flow_5_11_0_0
   + flow_8_5_0_0 + flow_10_5_0_0 = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_3_6_0_0
   + edge_4_6_0_0 + edge_5_6_0_0 + edge_8_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_3_6_0_0 + flow_4_6_0_0 + flow_5_6_0_0 - flow_6_2_0_0
   + flow_8_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_9_7_0_0
   + edge_10_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_4_0_0 - flow_7_8_0_0
   - flow_7_9_0_0 - flow_7_10_0_0 - flow_7_11_0_0 + flow_9_7_0_0
   + flow_10_7_0_0 = 1
 one_parent_8: edge_0_8_0_0 + edge_2_8_0_0 + edge_3_8_0_0 + edge_4_8_0_0
   + edge_5_8_0_0 + edge_7_8_0_0 + edge_10_8_0_0 = 1
 consume_one_flow_unit_8: flow_0_8_0_0 + flow_2_8_0_0 + flow_3_8_0_0
   + flow_4_8_0_0 + flow_5_8_0_0 + flow_7_8_0_0 - flow_8_2_0_0
   - flow_8_3_0_0 - flow_8_4_0_0 - flow_8_5_0_0 - flow_8_6_0_0
   - flow_8_9_0_0 - flow_8_10_0_0 - flow_8_11_0_0 + flow_10_8_0_0 = 1
 one_parent_9: edge_1_9_0_0 + edge_2_9_0_0 + edge_3_9_0_0 + edge_4_9_0_0
   + edge_5_9_0_0 + edge_7_9_0_0 + edge_8_9_0_0 + edge_10_9_0_0 = 1
 consume_one_flow_unit_9: flow_1_9_0_0 + flow_2_9_0_0 + flow_3_9_0_0
   + flow_4_9_0_0 + flow_5_9_0_0 + flow_7_9_0_0 + flow_8_9_0_0
   - flow_9_7_0_0 - flow_9_10_0_0 + flow_10_9_0_0 = 1
 one_parent_10: edge_2_10_0_0 + edge_4_10_0_0 + edge_7_10_0_0
   + edge_8_10_0_0 + edge_9_10_0_0 + edge_10_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_2_10_0_0 + flow_4_10_0_0 + flow_7_10_0_0
   + flow_8_10_0_0 + flow_9_10_0_0 - flow_10_1_0_0 - flow_10_2_0_0
   - flow_10_4_0_0 - flow_10_5_0_0 - flow_10_7_0_0 - flow_10_8_0_0
   - flow_10_9_0_0 - flow_10_11_0_0 + flow_11_10_0_0 = 1
 one_parent_11: edge_2_11_0_0 + edge_3_11_0_0 + edge_4_11_0_0
   + edge_5_11_0_0 + edge_7_11_0_0 + edge_8_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_2_11_0_0 + flow_3_11_0_0 + flow_4_11_0_0
   + flow_5_11_0_0 + flow_7_11_0_0 + flow_8_11_0_0 + flow_10_11_0_0
   - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_3_0_0 + flow_0_4_0_0
   + flow_0_5_0_0 + flow_0_6_0_0 + flow_0_7_0_0 + flow_0_8_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_3_0_0: flow_0_3_0_0 - 1000 edge_0_3_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_5_0_0: flow_0_5_0_0 - 1000 edge_0_5_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_8_0_0: flow_0_8_0_0 - 1000 edge_0_8_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
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
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_3_2_0_0: flow_3_2_0_0 - 1000 edge_3_2_0_0 <= 0
 inactive_no_flow_3_4_0_0: flow_3_4_0_0 - 1000 edge_3_4_0_0 <= 0
 inactive_no_flow_3_5_0_0: flow_3_5_0_0 - 1000 edge_3_5_0_0 <= 0
 inactive_no_flow_3_6_0_0: flow_3_6_0_0 - 1000 edge_3_6_0_0 <= 0
 inactive_no_flow_3_8_0_0: flow_3_8_0_0 - 1000 edge_3_8_0_0 <= 0
 inactive_no_flow_3_9_0_0: flow_3_9_0_0 - 1000 edge_3_9_0_0 <= 0
 inactive_no_flow_3_11_0_0: flow_3_11_0_0 - 1000 edge_3_11_0_0 <= 0
 inactive_no_flow_4_2_0_0: flow_4_2_0_0 - 1000 edge_4_2_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_5_0_0: flow_4_5_0_0 - 1000 edge_4_5_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_8_0_0: flow_4_8_0_0 - 1000 edge_4_8_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_4_11_0_0: flow_4_11_0_0 - 1000 edge_4_11_0_0 <= 0
 inactive_no_flow_5_2_0_0: flow_5_2_0_0 - 1000 edge_5_2_0_0 <= 0
 inactive_no_flow_5_3_0_0: flow_5_3_0_0 - 1000 edge_5_3_0_0 <= 0
 inactive_no_flow_5_4_0_0: flow_5_4_0_0 - 1000 edge_5_4_0_0 <= 0
 inactive_no_flow_5_5_0_0: flow_5_5_0_0 - 1000 edge_5_5_0_0 <= 0
 inactive_no_flow_5_6_0_0: flow_5_6_0_0 - 1000 edge_5_6_0_0 <= 0
 inactive_no_flow_5_8_0_0: flow_5_8_0_0 - 1000 edge_5_8_0_0 <= 0
 inactive_no_flow_5_9_0_0: flow_5_9_0_0 - 1000 edge_5_9_0_0 <= 0
 inactive_no_flow_5_11_0_0: flow_5_11_0_0 - 1000 edge_5_11_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_4_0_0: flow_7_4_0_0 - 1000 edge_7_4_0_0 <= 0
 inactive_no_flow_7_8_0_0: flow_7_8_0_0 - 1000 edge_7_8_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_8_2_0_0: flow_8_2_0_0 - 1000 edge_8_2_0_0 <= 0
 inactive_no_flow_8_3_0_0: flow_8_3_0_0 - 1000 edge_8_3_0_0 <= 0
 inactive_no_flow_8_4_0_0: flow_8_4_0_0 - 1000 edge_8_4_0_0 <= 0
 inactive_no_flow_8_5_0_0: flow_8_5_0_0 - 1000 edge_8_5_0_0 <= 0
 inactive_no_flow_8_6_0_0: flow_8_6_0_0 - 1000 edge_8_6_0_0 <= 0
 inactive_no_flow_8_9_0_0: flow_8_9_0_0 - 1000 edge_8_9_0_0 <= 0
 inactive_no_flow_8_10_0_0: flow_8_10_0_0 - 1000 edge_8_10_0_0 <= 0
 inactive_no_flow_8_11_0_0: flow_8_11_0_0 - 1000 edge_8_11_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_5_0_0: flow_10_5_0_0 - 1000 edge_10_5_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_8_0_0: flow_10_8_0_0 - 1000 edge_10_8_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_10_0_0: flow_10_10_0_0 - 1000 edge_10_10_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_3_0_0 edge_0_4_0_0 edge_0_5_0_0
 edge_0_6_0_0 edge_0_7_0_0 edge_0_8_0_0 edge_1_2_0_0 edge_1_6_0_0
 edge_1_7_0_0 edge_1_9_0_0 edge_2_1_0_0 edge_2_3_0_0 edge_2_4_0_0
 edge_2_5_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_8_0_0 edge_2_9_0_0
 edge_2_10_0_0 edge_2_11_0_0 edge_3_2_0_0 edge_3_4_0_0 edge_3_5_0_0
 edge_3_6_0_0 edge_3_8_0_0 edge_3_9_0_0 edge_3_11_0_0 edge_4_2_0_0
 edge_4_3_0_0 edge_4_5_0_0 edge_4_6_0_0 edge_4_8_0_0 edge_4_9_0_0
 edge_4_10_0_0 edge_4_11_0_0 edge_5_2_0_0 edge_5_3_0_0 edge_5_4_0_0
 edge_5_5_0_0 edge_5_6_0_0 edge_5_8_0_0 edge_5_9_0_0 edge_5_11_0_0
 edge_6_2_0_0 edge_7_1_0_0 edge_7_2_0_0 edge_7_4_0_0 edge_7_8_0_0
 edge_7_9_0_0 edge_7_10_0_0 edge_7_11_0_0 edge_8_2_0_0 edge_8_3_0_0
 edge_8_4_0_0 edge_8_5_0_0 edge_8_6_0_0 edge_8_9_0_0 edge_8_10_0_0
 edge_8_11_0_0 edge_9_7_0_0 edge_9_10_0_0 edge_10_1_0_0 edge_10_2_0_0
 edge_10_4_0_0 edge_10_5_0_0 edge_10_7_0_0 edge_10_8_0_0 edge_10_9_0_0
 edge_10_10_0_0 edge_10_11_0_0 edge_11_10_0_0
End
