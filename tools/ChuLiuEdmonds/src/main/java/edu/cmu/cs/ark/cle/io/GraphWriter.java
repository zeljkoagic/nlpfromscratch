package edu.cmu.cs.ark.cle.io;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class GraphWriter {
    private final PrintWriter writer;

    public GraphWriter(PrintWriter writer) {
        this.writer = writer;
    }

    public GraphWriter(File file) throws IOException {
        this(new PrintWriter(new BufferedWriter(new FileWriter(file))));
    }

    public GraphWriter(String fileName) throws IOException {
        this(new File(fileName));
    }

    public void writeGraph(ConllGraph graph) throws IOException {
        for(Integer i: graph.nodes.keySet()) {
            StringBuilder sb = new StringBuilder();
            sb.append(Integer.toString(i));
            sb.append("\t");
            sb.append(graph.nodes.get(i).form);
            sb.append("\t_\t");
            sb.append(graph.nodes.get(i).pos);
            sb.append("\t_\t_\t");
            sb.append(graph.nodes.get(i).head);
            sb.append("\t_\t_\t_");
            System.out.println(sb.toString());
        }
        
        System.out.println();
        
    }

}
