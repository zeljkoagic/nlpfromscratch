package edu.cmu.cs.ark.cle.io;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Reader;
import java.util.List;

import edu.cmu.cs.ark.cle.util.*;

public class GraphReader extends ParagraphReader {

    public GraphReader(Reader reader) {
        super(reader);
    }

    public GraphReader(File file) throws FileNotFoundException {
        super(file);
    }

    public GraphReader(String fileName) throws FileNotFoundException {
        super(fileName);
    }

    public ConllGraph readGraph() throws IOException {
        List<String> lines = super.readParagraph();
        ConllGraph graph = new ConllGraph();
        
        if (lines == null) {
            return null;
        } else {
            
            graph.matrix = new double[lines.size()+1][lines.size()+1];
            
            // column 0 are heads of ROOT
            for(int i = 0; i < lines.size(); ++ i) {
                graph.matrix[i][0] = 0.0;
            }
            
            assert lines.size() >= 2;
            
            for (String line : lines) {
                String[] tokens = line.split("\t| ");

                Integer id = Integer.valueOf(tokens[0]);
                String form = tokens[1];
                String pos = tokens[3];
                Integer head = Integer.valueOf(tokens[6]);
                String deprel = tokens[7];
                
                // fill in column ID (heads of token ID)
                for(int i = 10; i < tokens.length; ++ i) {
                    graph.matrix[i-10][id] = Double.valueOf(tokens[i]);
                }
                
                Node node = new Node(id, form, pos, head, deprel);
                graph.nodes.put(id, node);

                assert id == Integer.parseInt(tokens[0]);
            }
            
            return graph;
        }
    }
}
