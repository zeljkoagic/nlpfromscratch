Maximize
  - 2.7167713815056 edge_0_1_0_0 - 2.0405451857733 edge_0_2_0_0
   - 1.056209072464 edge_0_4_0_0 - 0.917169880284 edge_0_6_0_0
   - 2.505820613142 edge_0_7_0_0 - 1.95402552036 edge_0_9_0_0
   - 1.8279154492014 edge_0_10_0_0 - 4.763820973441001 edge_0_11_0_0
   + 0.7842455975302113 edge_1_2_0_0 + 0.0324486061033285 edge_1_3_0_0
   + 0.0901902394885685 edge_1_4_0_0 + 1.193806983075101 edge_1_6_0_0
   + 1.548183747594846 edge_1_7_0_0 + 0.5857826576255568 edge_1_9_0_0
   + 1.992531212900117 edge_1_10_0_0 + 2.452114208286449 edge_1_11_0_0
   + 0.1831138317744598 edge_2_1_0_0 + 0.4160004375199885 edge_2_6_0_0
   + 0.2850391317285378 edge_2_7_0_0 + 0.2891336778888173 edge_2_9_0_0
   + 0.397296547270524 edge_2_10_0_0 + 0.422962178759983 edge_2_11_0_0
   + 0.2208969325167967 edge_4_3_0_0 + 0.1900593616184957 edge_4_6_0_0
   - 0.1276870816714905 edge_4_9_0_0 + 0.0699037613547798 edge_4_10_0_0
   - 0.1414577064492906 edge_6_1_0_0 - 0.1340997301418428 edge_6_2_0_0
   + 0.0115843279108067 edge_6_3_0_0 + 0.0126299932278426 edge_6_4_0_0
   + 0.6951876576008021 edge_6_7_0_0 - 0.595426882805728 edge_6_9_0_0
   - 0.1329000217205188 edge_6_10_0_0 - 0.5297186397414198 edge_6_11_0_0
   - 1.176811732056441 edge_7_1_0_0 - 0.339227384848158 edge_7_2_0_0
   - 1.000417905479723 edge_7_6_0_0 - 0.3758222423341068 edge_7_9_0_0
   - 0.1464449369747454 edge_7_10_0_0 - 0.2068897840771322 edge_7_11_0_0
   + 0.2465014635793575 edge_9_1_0_0 + 0.2387391074679875 edge_9_2_0_0
   - 0.4190269652114003 edge_9_6_0_0 + 0.2636662756725285 edge_9_7_0_0
   + 1.890949253044789 edge_9_10_0_0 + 0.7275612765728188 edge_9_11_0_0
   + 0.2817873611901285 edge_10_1_0_0 + 0.5169543398922717 edge_10_2_0_0
   + 0.5970240917036475 edge_10_3_0_0 + 0.4697198352849857 edge_10_4_0_0
   + 0.8254173424179917 edge_10_6_0_0 + 0.9099308347123267 edge_10_7_0_0
   + 3.51144219690089 edge_10_9_0_0 + 0.1825512474759754 edge_10_10_0_0
   + 1.967577564832664 edge_10_11_0_0 - 1.804982256593358 edge_11_1_0_0
   - 0.6132726084353534 edge_11_2_0_0 - 1.850924087998304 edge_11_6_0_0
   - 0.5871515810134806 edge_11_7_0_0 - 0.7881392303125692 edge_11_9_0_0
   - 0.8617235538733611 edge_11_10_0_0
Subject To
 one_parent_1: edge_0_1_0_0 + edge_2_1_0_0 + edge_6_1_0_0 + edge_7_1_0_0
   + edge_9_1_0_0 + edge_10_1_0_0 + edge_11_1_0_0 = 1
 consume_one_flow_unit_1: flow_0_1_0_0 - flow_1_2_0_0 - flow_1_3_0_0
   - flow_1_4_0_0 - flow_1_6_0_0 - flow_1_7_0_0 - flow_1_9_0_0
   - flow_1_10_0_0 - flow_1_11_0_0 + flow_2_1_0_0 + flow_6_1_0_0
   + flow_7_1_0_0 + flow_9_1_0_0 + flow_10_1_0_0 + flow_11_1_0_0 = 1
 one_parent_2: edge_0_2_0_0 + edge_1_2_0_0 + edge_6_2_0_0 + edge_7_2_0_0
   + edge_9_2_0_0 + edge_10_2_0_0 + edge_11_2_0_0 = 1
 consume_one_flow_unit_2: flow_0_2_0_0 + flow_1_2_0_0 - flow_2_1_0_0
   - flow_2_6_0_0 - flow_2_7_0_0 - flow_2_9_0_0 - flow_2_10_0_0
   - flow_2_11_0_0 + flow_6_2_0_0 + flow_7_2_0_0 + flow_9_2_0_0
   + flow_10_2_0_0 + flow_11_2_0_0 = 1
 one_parent_3: edge_1_3_0_0 + edge_4_3_0_0 + edge_6_3_0_0 + edge_10_3_0_0
   = 1
 consume_one_flow_unit_3: flow_1_3_0_0 + flow_4_3_0_0 + flow_6_3_0_0
   + flow_10_3_0_0 = 1
 one_parent_4: edge_0_4_0_0 + edge_1_4_0_0 + edge_6_4_0_0 + edge_10_4_0_0
   = 1
 consume_one_flow_unit_4: flow_0_4_0_0 + flow_1_4_0_0 - flow_4_3_0_0
   - flow_4_6_0_0 - flow_4_9_0_0 - flow_4_10_0_0 + flow_6_4_0_0
   + flow_10_4_0_0 = 1
 one_parent_5: = 1
 consume_one_flow_unit_5: = 1
 one_parent_6: edge_0_6_0_0 + edge_1_6_0_0 + edge_2_6_0_0 + edge_4_6_0_0
   + edge_7_6_0_0 + edge_9_6_0_0 + edge_10_6_0_0 + edge_11_6_0_0 = 1
 consume_one_flow_unit_6: flow_0_6_0_0 + flow_1_6_0_0 + flow_2_6_0_0
   + flow_4_6_0_0 - flow_6_1_0_0 - flow_6_2_0_0 - flow_6_3_0_0
   - flow_6_4_0_0 - flow_6_7_0_0 - flow_6_9_0_0 - flow_6_10_0_0
   - flow_6_11_0_0 + flow_7_6_0_0 + flow_9_6_0_0 + flow_10_6_0_0
   + flow_11_6_0_0 = 1
 one_parent_7: edge_0_7_0_0 + edge_1_7_0_0 + edge_2_7_0_0 + edge_6_7_0_0
   + edge_9_7_0_0 + edge_10_7_0_0 + edge_11_7_0_0 = 1
 consume_one_flow_unit_7: flow_0_7_0_0 + flow_1_7_0_0 + flow_2_7_0_0
   + flow_6_7_0_0 - flow_7_1_0_0 - flow_7_2_0_0 - flow_7_6_0_0
   - flow_7_9_0_0 - flow_7_10_0_0 - flow_7_11_0_0 + flow_9_7_0_0
   + flow_10_7_0_0 + flow_11_7_0_0 = 1
 one_parent_8: = 1
 consume_one_flow_unit_8: = 1
 one_parent_9: edge_0_9_0_0 + edge_1_9_0_0 + edge_2_9_0_0 + edge_4_9_0_0
   + edge_6_9_0_0 + edge_7_9_0_0 + edge_10_9_0_0 + edge_11_9_0_0 = 1
 consume_one_flow_unit_9: flow_0_9_0_0 + flow_1_9_0_0 + flow_2_9_0_0
   + flow_4_9_0_0 + flow_6_9_0_0 + flow_7_9_0_0 - flow_9_1_0_0
   - flow_9_2_0_0 - flow_9_6_0_0 - flow_9_7_0_0 - flow_9_10_0_0
   - flow_9_11_0_0 + flow_10_9_0_0 + flow_11_9_0_0 = 1
 one_parent_10: edge_0_10_0_0 + edge_1_10_0_0 + edge_2_10_0_0
   + edge_4_10_0_0 + edge_6_10_0_0 + edge_7_10_0_0 + edge_9_10_0_0
   + edge_10_10_0_0 + edge_11_10_0_0 = 1
 consume_one_flow_unit_10: flow_0_10_0_0 + flow_1_10_0_0 + flow_2_10_0_0
   + flow_4_10_0_0 + flow_6_10_0_0 + flow_7_10_0_0 + flow_9_10_0_0
   - flow_10_1_0_0 - flow_10_2_0_0 - flow_10_3_0_0 - flow_10_4_0_0
   - flow_10_6_0_0 - flow_10_7_0_0 - flow_10_9_0_0 - flow_10_11_0_0
   + flow_11_10_0_0 = 1
 one_parent_11: edge_0_11_0_0 + edge_1_11_0_0 + edge_2_11_0_0
   + edge_6_11_0_0 + edge_7_11_0_0 + edge_9_11_0_0 + edge_10_11_0_0 = 1
 consume_one_flow_unit_11: flow_0_11_0_0 + flow_1_11_0_0 + flow_2_11_0_0
   + flow_6_11_0_0 + flow_7_11_0_0 + flow_9_11_0_0 + flow_10_11_0_0
   - flow_11_1_0_0 - flow_11_2_0_0 - flow_11_6_0_0 - flow_11_7_0_0
   - flow_11_9_0_0 - flow_11_10_0_0 = 1
 root_flow: flow_0_1_0_0 + flow_0_2_0_0 + flow_0_4_0_0 + flow_0_6_0_0
   + flow_0_7_0_0 + flow_0_9_0_0 + flow_0_10_0_0 + flow_0_11_0_0 = 11
 inactive_no_flow_0_1_0_0: flow_0_1_0_0 - 1000 edge_0_1_0_0 <= 0
 inactive_no_flow_0_2_0_0: flow_0_2_0_0 - 1000 edge_0_2_0_0 <= 0
 inactive_no_flow_0_4_0_0: flow_0_4_0_0 - 1000 edge_0_4_0_0 <= 0
 inactive_no_flow_0_6_0_0: flow_0_6_0_0 - 1000 edge_0_6_0_0 <= 0
 inactive_no_flow_0_7_0_0: flow_0_7_0_0 - 1000 edge_0_7_0_0 <= 0
 inactive_no_flow_0_9_0_0: flow_0_9_0_0 - 1000 edge_0_9_0_0 <= 0
 inactive_no_flow_0_10_0_0: flow_0_10_0_0 - 1000 edge_0_10_0_0 <= 0
 inactive_no_flow_0_11_0_0: flow_0_11_0_0 - 1000 edge_0_11_0_0 <= 0
 inactive_no_flow_1_2_0_0: flow_1_2_0_0 - 1000 edge_1_2_0_0 <= 0
 inactive_no_flow_1_3_0_0: flow_1_3_0_0 - 1000 edge_1_3_0_0 <= 0
 inactive_no_flow_1_4_0_0: flow_1_4_0_0 - 1000 edge_1_4_0_0 <= 0
 inactive_no_flow_1_6_0_0: flow_1_6_0_0 - 1000 edge_1_6_0_0 <= 0
 inactive_no_flow_1_7_0_0: flow_1_7_0_0 - 1000 edge_1_7_0_0 <= 0
 inactive_no_flow_1_9_0_0: flow_1_9_0_0 - 1000 edge_1_9_0_0 <= 0
 inactive_no_flow_1_10_0_0: flow_1_10_0_0 - 1000 edge_1_10_0_0 <= 0
 inactive_no_flow_1_11_0_0: flow_1_11_0_0 - 1000 edge_1_11_0_0 <= 0
 inactive_no_flow_2_1_0_0: flow_2_1_0_0 - 1000 edge_2_1_0_0 <= 0
 inactive_no_flow_2_6_0_0: flow_2_6_0_0 - 1000 edge_2_6_0_0 <= 0
 inactive_no_flow_2_7_0_0: flow_2_7_0_0 - 1000 edge_2_7_0_0 <= 0
 inactive_no_flow_2_9_0_0: flow_2_9_0_0 - 1000 edge_2_9_0_0 <= 0
 inactive_no_flow_2_10_0_0: flow_2_10_0_0 - 1000 edge_2_10_0_0 <= 0
 inactive_no_flow_2_11_0_0: flow_2_11_0_0 - 1000 edge_2_11_0_0 <= 0
 inactive_no_flow_4_3_0_0: flow_4_3_0_0 - 1000 edge_4_3_0_0 <= 0
 inactive_no_flow_4_6_0_0: flow_4_6_0_0 - 1000 edge_4_6_0_0 <= 0
 inactive_no_flow_4_9_0_0: flow_4_9_0_0 - 1000 edge_4_9_0_0 <= 0
 inactive_no_flow_4_10_0_0: flow_4_10_0_0 - 1000 edge_4_10_0_0 <= 0
 inactive_no_flow_6_1_0_0: flow_6_1_0_0 - 1000 edge_6_1_0_0 <= 0
 inactive_no_flow_6_2_0_0: flow_6_2_0_0 - 1000 edge_6_2_0_0 <= 0
 inactive_no_flow_6_3_0_0: flow_6_3_0_0 - 1000 edge_6_3_0_0 <= 0
 inactive_no_flow_6_4_0_0: flow_6_4_0_0 - 1000 edge_6_4_0_0 <= 0
 inactive_no_flow_6_7_0_0: flow_6_7_0_0 - 1000 edge_6_7_0_0 <= 0
 inactive_no_flow_6_9_0_0: flow_6_9_0_0 - 1000 edge_6_9_0_0 <= 0
 inactive_no_flow_6_10_0_0: flow_6_10_0_0 - 1000 edge_6_10_0_0 <= 0
 inactive_no_flow_6_11_0_0: flow_6_11_0_0 - 1000 edge_6_11_0_0 <= 0
 inactive_no_flow_7_1_0_0: flow_7_1_0_0 - 1000 edge_7_1_0_0 <= 0
 inactive_no_flow_7_2_0_0: flow_7_2_0_0 - 1000 edge_7_2_0_0 <= 0
 inactive_no_flow_7_6_0_0: flow_7_6_0_0 - 1000 edge_7_6_0_0 <= 0
 inactive_no_flow_7_9_0_0: flow_7_9_0_0 - 1000 edge_7_9_0_0 <= 0
 inactive_no_flow_7_10_0_0: flow_7_10_0_0 - 1000 edge_7_10_0_0 <= 0
 inactive_no_flow_7_11_0_0: flow_7_11_0_0 - 1000 edge_7_11_0_0 <= 0
 inactive_no_flow_9_1_0_0: flow_9_1_0_0 - 1000 edge_9_1_0_0 <= 0
 inactive_no_flow_9_2_0_0: flow_9_2_0_0 - 1000 edge_9_2_0_0 <= 0
 inactive_no_flow_9_6_0_0: flow_9_6_0_0 - 1000 edge_9_6_0_0 <= 0
 inactive_no_flow_9_7_0_0: flow_9_7_0_0 - 1000 edge_9_7_0_0 <= 0
 inactive_no_flow_9_10_0_0: flow_9_10_0_0 - 1000 edge_9_10_0_0 <= 0
 inactive_no_flow_9_11_0_0: flow_9_11_0_0 - 1000 edge_9_11_0_0 <= 0
 inactive_no_flow_10_1_0_0: flow_10_1_0_0 - 1000 edge_10_1_0_0 <= 0
 inactive_no_flow_10_2_0_0: flow_10_2_0_0 - 1000 edge_10_2_0_0 <= 0
 inactive_no_flow_10_3_0_0: flow_10_3_0_0 - 1000 edge_10_3_0_0 <= 0
 inactive_no_flow_10_4_0_0: flow_10_4_0_0 - 1000 edge_10_4_0_0 <= 0
 inactive_no_flow_10_6_0_0: flow_10_6_0_0 - 1000 edge_10_6_0_0 <= 0
 inactive_no_flow_10_7_0_0: flow_10_7_0_0 - 1000 edge_10_7_0_0 <= 0
 inactive_no_flow_10_9_0_0: flow_10_9_0_0 - 1000 edge_10_9_0_0 <= 0
 inactive_no_flow_10_10_0_0: flow_10_10_0_0 - 1000 edge_10_10_0_0 <= 0
 inactive_no_flow_10_11_0_0: flow_10_11_0_0 - 1000 edge_10_11_0_0 <= 0
 inactive_no_flow_11_1_0_0: flow_11_1_0_0 - 1000 edge_11_1_0_0 <= 0
 inactive_no_flow_11_2_0_0: flow_11_2_0_0 - 1000 edge_11_2_0_0 <= 0
 inactive_no_flow_11_6_0_0: flow_11_6_0_0 - 1000 edge_11_6_0_0 <= 0
 inactive_no_flow_11_7_0_0: flow_11_7_0_0 - 1000 edge_11_7_0_0 <= 0
 inactive_no_flow_11_9_0_0: flow_11_9_0_0 - 1000 edge_11_9_0_0 <= 0
 inactive_no_flow_11_10_0_0: flow_11_10_0_0 - 1000 edge_11_10_0_0 <= 0
Bounds
Binaries
 edge_0_1_0_0 edge_0_2_0_0 edge_0_4_0_0 edge_0_6_0_0 edge_0_7_0_0
 edge_0_9_0_0 edge_0_10_0_0 edge_0_11_0_0 edge_1_2_0_0 edge_1_3_0_0
 edge_1_4_0_0 edge_1_6_0_0 edge_1_7_0_0 edge_1_9_0_0 edge_1_10_0_0
 edge_1_11_0_0 edge_2_1_0_0 edge_2_6_0_0 edge_2_7_0_0 edge_2_9_0_0
 edge_2_10_0_0 edge_2_11_0_0 edge_4_3_0_0 edge_4_6_0_0 edge_4_9_0_0
 edge_4_10_0_0 edge_6_1_0_0 edge_6_2_0_0 edge_6_3_0_0 edge_6_4_0_0
 edge_6_7_0_0 edge_6_9_0_0 edge_6_10_0_0 edge_6_11_0_0 edge_7_1_0_0
 edge_7_2_0_0 edge_7_6_0_0 edge_7_9_0_0 edge_7_10_0_0 edge_7_11_0_0
 edge_9_1_0_0 edge_9_2_0_0 edge_9_6_0_0 edge_9_7_0_0 edge_9_10_0_0
 edge_9_11_0_0 edge_10_1_0_0 edge_10_2_0_0 edge_10_3_0_0 edge_10_4_0_0
 edge_10_6_0_0 edge_10_7_0_0 edge_10_9_0_0 edge_10_10_0_0 edge_10_11_0_0
 edge_11_1_0_0 edge_11_2_0_0 edge_11_6_0_0 edge_11_7_0_0 edge_11_9_0_0
 edge_11_10_0_0
End
