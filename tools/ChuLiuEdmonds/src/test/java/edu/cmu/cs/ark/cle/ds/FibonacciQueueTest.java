package edu.cmu.cs.ark.cle.ds;

import com.google.common.collect.*;
import org.junit.Test;

import java.util.List;
import java.util.Random;
import java.util.Set;

import static com.google.common.collect.DiscreteDomain.integers;
import static com.google.common.collect.Range.closedOpen;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class FibonacciQueueTest {
	@Test
	public void testIterator() {
		// insert lots of numbers in order
		final Set<Integer> values = ContiguousSet.create(closedOpen(0, 1000), integers());
		final FibonacciQueue<Integer> queue = FibonacciQueue.create();
		assertTrue(queue.addAll(values));
		assertEquals(values, ImmutableSet.copyOf(queue.iterator()));
		assertEquals(values, ImmutableSet.copyOf(queue));
	}

	@Test
	public void testLotsOfRandomInserts() {
		int lots = 50000;
		final FibonacciQueue<Integer> queue = FibonacciQueue.create();
		// Insert lots of random numbers.
		final ImmutableMultiset.Builder<Integer> insertedBuilder = ImmutableMultiset.builder();
		final Random random = new Random();
		for (int i = 0; i < lots; i++) {
			int r = random.nextInt();
			insertedBuilder.add(r);
			queue.add(r);
		}
		final Multiset<Integer> inserted = insertedBuilder.build();
		assertEquals(lots, queue.size());
		// Ensure it contains the same multiset of values that we put in
		assertEquals(inserted, ImmutableMultiset.copyOf(queue));
		// Ensure the numbers come out in increasing order.
		final List<Integer> polled = Lists.newLinkedList();
		while (!queue.isEmpty()) {
			polled.add(queue.poll());
		}
		assertTrue(Ordering.<Integer>natural().isOrdered(polled));
		// Ensure the same multiset of values came out that we put in
		assertEquals(inserted, ImmutableMultiset.copyOf(polled));
		assertEquals(0, queue.size());
	}
}
