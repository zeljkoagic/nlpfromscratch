/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.cmu.cs.ark.cle;

import edu.cmu.cs.ark.cle.graph.*;
import edu.cmu.cs.ark.cle.util.*;
import edu.cmu.cs.ark.cle.io.*;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 *
 * @author zagic
 */
public class Test {

    public static void main(String[] args) throws Exception {
        
        GraphReader reader = new GraphReader(new InputStreamReader(System.in));
        GraphWriter writer = new GraphWriter(new PrintWriter(System.out));
        
        ConllGraph graph;

        // read graph
        while ((graph = reader.readGraph()) != null) {
        
            // run MST
            DenseWeightedGraph dwg = DenseWeightedGraph.from(graph.matrix);
            Weighted x = ChuLiuEdmonds.getMaxArborescence(dwg, 0);
            Arborescence<Integer> a = (Arborescence) x.val;
            
            //System.out.println(dwg.getNodes());
            //System.out.println(a);
            
            // apply heads to nodes
            for(Node node: graph.nodes.values()) {
                node.head = a.parents.get(node.id);
            }
            
            // print graph
            writer.writeGraph(graph);
        }

        reader.close();
    }
}
