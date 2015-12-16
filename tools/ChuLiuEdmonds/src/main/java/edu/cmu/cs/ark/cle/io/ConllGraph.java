package edu.cmu.cs.ark.cle.io;

import java.util.Map;
import java.util.HashMap;
import edu.cmu.cs.ark.cle.util.Node;

public class ConllGraph {
    public Map<Integer, Node> nodes;
    public double[][] matrix;
    public ConllGraph(){
        nodes = new HashMap<Integer, Node>();
    }
}
