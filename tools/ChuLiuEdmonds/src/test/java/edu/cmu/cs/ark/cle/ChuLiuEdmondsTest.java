package edu.cmu.cs.ark.cle;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import edu.cmu.cs.ark.cle.graph.DenseWeightedGraph;
import edu.cmu.cs.ark.cle.graph.Edge;
import edu.cmu.cs.ark.cle.graph.SparseWeightedGraph;
import edu.cmu.cs.ark.cle.graph.WeightedGraph;
import edu.cmu.cs.ark.cle.util.Weighted;
import org.junit.Test;

import java.util.Map;

import static edu.cmu.cs.ark.cle.util.Weighted.weighted;
import static org.junit.Assert.assertEquals;

/**
 * @author sthomson@cs.cmu.edu
 */
public class ChuLiuEdmondsTest {
	final static double DELTA = 0.001;
	final static double NINF = Double.NEGATIVE_INFINITY;
	final static WeightedGraph<Integer> graph = SparseWeightedGraph.from(ImmutableList.of(
			weighted(Edge.from(0).to(1), 5),
			weighted(Edge.from(0).to(2), 1),
			weighted(Edge.from(0).to(3), 1),
			weighted(Edge.from(1).to(2), 11),
			weighted(Edge.from(1).to(3), 4),
			weighted(Edge.from(2).to(1), 10),
			weighted(Edge.from(2).to(3), 5),
			weighted(Edge.from(3).to(1), 9),
			weighted(Edge.from(3).to(2), 8)
	));

	static <V> void assertEdgesSumToScore(WeightedGraph<V> originalEdgeWeights, Weighted<Arborescence<V>> bestTree) {
		final Map<V, V> parentsMap = bestTree.val.parents;
		double sumOfWeights = 0.0;
		for (V dest : parentsMap.keySet()) {
			final V source = parentsMap.get(dest);
			sumOfWeights += originalEdgeWeights.getWeightOf(source, dest);
		}
		assertEquals(sumOfWeights, bestTree.weight, DELTA);
	}

	@Test
	public void testGetMaxSpanningTree() {
		/*
		root    10
		(0) -------> (1) \
		 |  \       /  ^  \
		 |   \30   |   |20 \
		 |10  \    |10 |    \10
		 |     \   |  /      \
		 V  15  V  V /   20   V
		(3)<----- (2) -----> (4)
		  \-------^
		     40
		 */
		double[][] weights = {
				{NINF, 10, 30, 10, NINF},
				{NINF, NINF, 10, NINF, 10 },
				{NINF,  20, NINF,  7, 20 },
				{NINF, NINF, 40, NINF, NINF},
				{NINF, NINF, NINF, NINF, NINF},
		};
		final DenseWeightedGraph<Integer> graph = DenseWeightedGraph.from(weights);
		final Weighted<Arborescence<Integer>> weightedSpanningTree = ChuLiuEdmonds.getMaxArborescence(graph, 0);
		/*
		root
		(0)           (1)
		 |             ^
		 |             |
		 |             |
		 |            /
		 V           /
		(3)       (2) ------> (4)
		  \-------^
		 */
		final Map<Integer, Integer> maxBranching = weightedSpanningTree.val.parents;
		assertEquals(2, maxBranching.get(1).intValue());
		assertEquals(3, maxBranching.get(2).intValue());
		assertEquals(0, maxBranching.get(3).intValue());
		assertEquals(2, maxBranching.get(4).intValue());
		assertEquals(90.0, weightedSpanningTree.weight, DELTA);
		assertEdgesSumToScore(graph, weightedSpanningTree);
	}

	@Test
	public void testRequiredAndBannedEdges() {
		final Weighted<Arborescence<Integer>> weightedSpanningTree = ChuLiuEdmonds.getMaxArborescence(
				graph,
				ImmutableSet.of(Edge.from(0).to(1)),
				ImmutableSet.of(Edge.from(2).to(3)));
		final Map<Integer, Integer> maxBranching = weightedSpanningTree.val.parents;
		assertEquals(0, maxBranching.get(1).intValue());
		assertEquals(1, maxBranching.get(2).intValue());
		assertEquals(1, maxBranching.get(3).intValue());
		assertEquals(20.0, weightedSpanningTree.weight, DELTA);
		assertEdgesSumToScore(graph, weightedSpanningTree);

	}

	@Test
	public void testRequiredAndBannedEdges2() {
		final Weighted<Arborescence<Integer>> weightedSpanningTree = ChuLiuEdmonds.getMaxArborescence(
				graph,
				ImmutableSet.of(Edge.from(0).to(3), Edge.from(3).to(1)),
				ImmutableSet.of(Edge.from(1).to(2))
		);
		final Map<Integer, Integer> maxBranching = weightedSpanningTree.val.parents;
		assertEquals(3, maxBranching.get(1).intValue());
		assertEquals(3, maxBranching.get(2).intValue());
		assertEquals(0, maxBranching.get(3).intValue());
		assertEquals(18.0, weightedSpanningTree.weight, DELTA);
		assertEdgesSumToScore(graph, weightedSpanningTree);

	}

	@Test
	public void testElevenNodeGraph() {
		// make a graph with a bunch of nested cycles so we can exercise the recursive part of the algorithm.
		final WeightedGraph<Integer> graph = SparseWeightedGraph.from(ImmutableList.of(
				weighted(Edge.from(0).to(8), 0),
				weighted(Edge.from(1).to(2), 10),
				weighted(Edge.from(1).to(4), 5),
				weighted(Edge.from(2).to(3), 9),
				weighted(Edge.from(3).to(1), 8),
				weighted(Edge.from(4).to(5), 9),
				weighted(Edge.from(5).to(6), 10),
				weighted(Edge.from(6).to(4), 8),
				weighted(Edge.from(6).to(7), 5),
				weighted(Edge.from(7).to(8), 10),
				weighted(Edge.from(8).to(2), 5),
				weighted(Edge.from(8).to(9), 8),
				weighted(Edge.from(8).to(10), 1),
				weighted(Edge.from(9).to(7), 9),
				weighted(Edge.from(10).to(3), 3)
		));
		final Weighted<Arborescence<Integer>> weightedSpanningTree = ChuLiuEdmonds.getMaxArborescence(graph, 0);

		final Map<Integer, Integer> maxBranching = weightedSpanningTree.val.parents;
		assertEdgesSumToScore(graph, weightedSpanningTree);
		assertEquals(3, maxBranching.get(1).intValue());
		assertEquals(8, maxBranching.get(2).intValue());
		assertEquals(2, maxBranching.get(3).intValue());
		assertEquals(1, maxBranching.get(4).intValue());
		assertEquals(4, maxBranching.get(5).intValue());
		assertEquals(5, maxBranching.get(6).intValue());
		assertEquals(9, maxBranching.get(7).intValue());
		assertEquals(0, maxBranching.get(8).intValue());
		assertEquals(8, maxBranching.get(9).intValue());
		assertEquals(8, maxBranching.get(10).intValue());
	}
}
