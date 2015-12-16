package edu.cmu.cs.ark.cle;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import edu.cmu.cs.ark.cle.util.Pair;
import edu.cmu.cs.ark.cle.ds.Partition;
import edu.cmu.cs.ark.cle.graph.Edge;
import edu.cmu.cs.ark.cle.graph.WeightedGraph;
import edu.cmu.cs.ark.cle.util.Weighted;

import java.util.*;

import static com.google.common.base.Predicates.and;
import static com.google.common.base.Predicates.not;
import static edu.cmu.cs.ark.cle.EdgeQueueMap.EdgeQueue;
import static edu.cmu.cs.ark.cle.util.Weighted.weighted;

/**
 * Chu-Liu-Edmonds' algorithm for finding a maximum branching in a complete, directed graph in O(n^2) time.
 * Implementation is based on Tarjan's "Finding Optimum Branchings" paper:
 * http://cw.felk.cvut.cz/lib/exe/fetch.php/courses/a4m33pal/cviceni/tarjan-finding-optimum-branchings.pdf
 *
 * @author sthomson@cs.cmu.edu
 */
public class ChuLiuEdmonds {
	/** Represents the subgraph that gets iteratively built up in the CLE algorithm. */
	static class PartialSolution<V> {
		// Partition representing the strongly connected components (SCCs).
		private final Partition<V> stronglyConnected;
		// Partition representing the weakly connected components (WCCs).
		private final Partition<V> weaklyConnected;
		// An invariant of the CLE algorithm is that each SCC always has at most one incoming edge.
		// You can think of these edges as implicitly defining a graph with SCCs as nodes.
		private final Map<V, Weighted<Edge<V>>> incomingEdgeByScc;
		// History of edges we've added, and for each, a list of edges it would exclude.
		// More recently added edges get priority over less recently added edges when reconstructing the final tree.
		private final LinkedList<ExclusiveEdge<V>> edgesAndWhatTheyExclude;
		// a priority queue of incoming edges for each SCC that we haven't chosen an incoming edge for yet.
		final EdgeQueueMap<V> unseenIncomingEdges;
		// running sum of weights.
		// edge weights are adjusted as we go to take into account the fact that we have an extra edge in each cycle
		private double score;

		private PartialSolution(Partition<V> stronglyConnected,
								Partition<V> weaklyConnected,
								Map<V, Weighted<Edge<V>>> incomingEdgeByScc,
								LinkedList<ExclusiveEdge<V>> edgesAndWhatTheyExclude,
								EdgeQueueMap<V> unseenIncomingEdges,
								double score) {
			this.stronglyConnected = stronglyConnected;
			this.weaklyConnected = weaklyConnected;
			this.incomingEdgeByScc = incomingEdgeByScc;
			this.edgesAndWhatTheyExclude = edgesAndWhatTheyExclude;
			this.unseenIncomingEdges = unseenIncomingEdges;
			this.score = score;
		}

		public static <T> PartialSolution<T> initialize(WeightedGraph<T> graph) {
			final Partition<T> stronglyConnected = Partition.singletons(graph.getNodes());
			final HashMap<T, Weighted<Edge<T>>> incomingByScc = Maps.newHashMap();
			final LinkedList<ExclusiveEdge<T>> exclusiveEdges = Lists.newLinkedList();
			// group edges by their destination component
			final EdgeQueueMap<T> incomingEdges = new EdgeQueueMap<T>(stronglyConnected);
			for (T destinationNode : graph.getNodes()) {
				for (Weighted<Edge<T>> inEdge : graph.getIncomingEdges(destinationNode)) {
					if (inEdge.weight != Double.NEGATIVE_INFINITY) {
						incomingEdges.addEdge(inEdge);
					}
				}
			}
			return new PartialSolution<T>(
					stronglyConnected,
					Partition.singletons(graph.getNodes()),
					incomingByScc,
					exclusiveEdges,
					incomingEdges,
					0.0
			);
		}

		public Set<V> getNodes() {
			return stronglyConnected.getNodes();
		}

		/**
		 * Given an edge that completes a cycle, merge all SCCs on that cycle into one SCC.
		 * Returns the new component.
		 */
		private V merge(Weighted<Edge<V>> newEdge, EdgeQueueMap<V> unseenIncomingEdges) {
			// Find edges connecting SCCs on the path from newEdge.destination to newEdge.source
			final List<Weighted<Edge<V>>> cycle = getCycle(newEdge);
			// build up list of queues that need to be merged, with the edge they would exclude
			final List<Pair<EdgeQueue<V>, Weighted<Edge<V>>>> queuesToMerge = Lists.newLinkedList();
			for (Weighted<Edge<V>> currentEdge : cycle) {
				final V destination = stronglyConnected.componentOf(currentEdge.val.destination);
				final EdgeQueue<V> queue = unseenIncomingEdges.queueByDestination.get(destination);
				// if we choose an edge in `queue`, we'll have to throw out `currentEdge` at the end
				// (each SCC can have only one incoming edge).
				queuesToMerge.add(Pair.of(queue, currentEdge));
				unseenIncomingEdges.queueByDestination.remove(destination);
			}
			// Merge all SCCs on the cycle into one
			for (Weighted<Edge<V>> e : cycle) {
				stronglyConnected.merge(e.val.source, e.val.destination);
			}
			V component = stronglyConnected.componentOf(newEdge.val.destination);
			// merge the queues and put the merged queue back into our map under the new component
			unseenIncomingEdges.merge(component, queuesToMerge);
			// keep our implicit graph of SCCs up to date:
			// we just created a cycle, so all in-edges have sources inside the new component
			// i.e. there is no edge with source outside component, and destination inside component
			incomingEdgeByScc.remove(component);
			return component;
		}

		/** Gets the cycle of edges between SCCs that newEdge creates */
		private List<Weighted<Edge<V>>> getCycle(Weighted<Edge<V>> newEdge) {
			final List<Weighted<Edge<V>>> cycle = Lists.newLinkedList();
			// circle around backward until you get back to where you started
			Weighted<Edge<V>> edge = newEdge;
			cycle.add(edge);
			while (!stronglyConnected.sameComponent(edge.val.source, newEdge.val.destination)) {
				edge = incomingEdgeByScc.get(stronglyConnected.componentOf(edge.val.source));
				cycle.add(edge);
			}
			return cycle;
		}

		/**
		 * Adds the given edge to this subgraph, merging SCCs if necessary
		 * @return the new SCC if adding edge created a cycle
		 */
		public Optional<V> addEdge(ExclusiveEdge<V> wEdgeAndExcludes) {
			final Edge<V> edge = wEdgeAndExcludes.edge;
			final double weight = wEdgeAndExcludes.weight;
			final Weighted<Edge<V>> wEdge = weighted(edge, weight);
			score += weight;
			final V destinationScc = stronglyConnected.componentOf(edge.destination);
			edgesAndWhatTheyExclude.addFirst(wEdgeAndExcludes);
			incomingEdgeByScc.put(destinationScc, wEdge);
			if (!weaklyConnected.sameComponent(edge.source, edge.destination)) {
				// Edge connects two different WCCs. Including it won't create a new cycle
				weaklyConnected.merge(edge.source, edge.destination);
				return Optional.absent();
			} else {
				// Edge is contained within one WCC. Including it will create a new cycle.
				return Optional.of(merge(wEdge, unseenIncomingEdges));
			}
		}

		/**
		 * Recovers the optimal arborescence.
		 *
		 * Each SCC can only have 1 edge entering it: the edge that we added most recently.
		 * So we work backwards, adding edges unless they conflict with edges we've already added.
		 * Runtime is O(n^2) in the worst case.
		 */
		private Weighted<Arborescence<V>> recoverBestArborescence() {
			final ImmutableMap.Builder<V, V> parents = ImmutableMap.builder();
			final Set<Edge> excluded = Sets.newHashSet();
			// start with the most recent
			while (!edgesAndWhatTheyExclude.isEmpty()) {
				final ExclusiveEdge<V> edgeAndWhatItExcludes = edgesAndWhatTheyExclude.pollFirst();
				final Edge<V> edge = edgeAndWhatItExcludes.edge;
				if(!excluded.contains(edge)) {
					excluded.addAll(edgeAndWhatItExcludes.excluded);
					parents.put(edge.destination, edge.source);
				}
			}
			return weighted(Arborescence.of(parents.build()), score);
		}

		public Optional<ExclusiveEdge<V>> popBestEdge(V component) {
			return popBestEdge(component, Arborescence.<V>empty());
		}

		/** Always breaks ties in favor of edges in `best` */
		public Optional<ExclusiveEdge<V>> popBestEdge(V component, Arborescence<V> best) {
			return unseenIncomingEdges.popBestEdge(component, best);
		}
	}

	/**
	 * Find an optimal arborescence of the given graph `graph`, rooted in the given node `root`.
	 */
	public static <V> Weighted<Arborescence<V>> getMaxArborescence(WeightedGraph<V> graph, V root) {
		// remove all edges incoming to `root`. resulting arborescence is then forced to be rooted at `root`.
		return getMaxArborescence(graph.filterEdges(not(Edge.hasDestination(root))));
	}

	static <V> Weighted<Arborescence<V>> getMaxArborescence(WeightedGraph<V> graph,
															Set<Edge<V>> required,
															Set<Edge<V>> banned) {
		return getMaxArborescence(graph.filterEdges(and(not(Edge.competesWith(required)), not(Edge.isIn(banned)))));
	}

	/**
	 * Find an optimal arborescence of the given graph.
	 */
	public static <V> Weighted<Arborescence<V>> getMaxArborescence(WeightedGraph<V> graph) {
		final PartialSolution<V> partialSolution =
				PartialSolution.initialize(graph.filterEdges(not(Edge.<V>isAutoCycle())));
		// In the beginning, subgraph has no edges, so no SCC has in-edges.
		final Queue<V> componentsWithNoInEdges = Lists.newLinkedList(partialSolution.getNodes());

		// Work our way through all componentsWithNoInEdges, in no particular order
		while (!componentsWithNoInEdges.isEmpty()) {
			final V component = componentsWithNoInEdges.poll();
			// find maximum edge entering 'component' from outside 'component'.
			final Optional<ExclusiveEdge<V>> oMaxInEdge = partialSolution.popBestEdge(component);
			if (!oMaxInEdge.isPresent()) continue; // No in-edges left to consider for this component. Done with it!
			final ExclusiveEdge<V> maxInEdge = oMaxInEdge.get();
			// add the new edge to subgraph, merging SCCs if necessary
			final Optional<V> newComponent = partialSolution.addEdge(maxInEdge);
			if (newComponent.isPresent()) {
				// addEdge created a cycle/component, which means the new component doesn't have any incoming edges
				componentsWithNoInEdges.add(newComponent.get());
			}
		}
		// Once no component has incoming edges left to consider, it's time to recover the optimal branching.
		return partialSolution.recoverBestArborescence();
	}
}
