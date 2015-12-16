package edu.cmu.cs.ark.cle.util;

/**
 * A node in a semantic dependency graph.
 *
 * @author Zeljko Agic <zagic@uni-potsdam.de>
 */
public class Node {

    public final Integer id;
    public final String form;
    public final String pos;
    public Integer head;
    public String deprel;
    
    public Node(int id, String form, String pos, int head, String deprel) {
        this.id = id;
        this.form = form;
        this.pos = pos;
        this.head = head;
        this.deprel = deprel;
    }
}