Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Remark (Research ongoing, but now slowly converging):

- To save training effort, a vectorization following a greedy approach was implemented, that bundles calculations of individual inference rules based on which can perform the most in an instant.

- Rule 52 (unit resulting resolution) does not lead to a selected node in any training instance used for testing (20,000 from unguided Vampire run with 10s time limit). And it's awfully slow to compute. It's banned for now, i.e., if new training instances arrive that contain a selected node which is derived by rule 52, it will still not be computed.

- The guidance helps to prove theorems if enough clauses are rated negatively to prevent combinatoric explosion, while not making a single mistake - clauses required for proofs are believed to be almost never replaceable. To shed insight on how generalization works, several "Zero"-Tests are being performed, which have a determined behavior, instead of a fitted one by training:
  1) Lookup clauses from solved problems (training set) and put positive / negative based on this experience. For clauses not seen in training, put positive. Result (preliminary): Stunningly, <1% deviation in number of solved problems compared to unguided Vampire. This shows, that the negatively rated clauses from the training set either interfere with other problems, or aren't enough to prevent combinatoric explosion. Since 90% of the negatively rated clauses are also negatively rated in the neural model, where 25% more problems are solved (instead of <1%), we conclude that the governing argument is that they aren't enough to prevent combinatoric explosion.
  2) To be done when 1) is finished: 1) Suggests, that the "unknown" axiom approach by Martin Suda has good generalization properties in a neighborhood of 25%-40% of the training data. Thereby we mean, that the rating of derivations from unseen axioms in about 25%-40% further problems is statistically resembled by derivations of less common axioms present in the training set and blanked out for this purpose. Therefore, we will do another "Zero"-Test: Consider the full merged multi-tree of all training data. For every problem, set those axioms to 0, which are not present in the specific problem, and also draw together the corresponding derivations. Guide the derivations of the axioms that are present in the problem but not in the training data by the former and corresponding derivations. We need to add a rule for derivations which are still unseen to obtain an algorithm. TODO (Maybe just positive in the first instance again?)

If 2) above shows good generalization, we will argue that the generalization of Martin Suda's architecture relies on the fact, that in the regime of the given training data, many small-medium sized problems have derivations that can be observed by the "unknown" axiom method (at not too high tree heights, it is often times the case that unseen derivations can be resembled from "unknown" axioms). For example a binary rule applied to 0 and 0 is rated positively. We yet know just of this compelling positive example, and need to look up a few understandable negative ones. For problems not solved by the guidance, probably many more more correct negative ratings are required, and the generalization of the model to "uncharted terrain" is bad. 

- Modifying evaluation and inference rules to have height-dependent tensor layers instead of a single fixed one don't improve the result - in contrast to expectation.  

As a background info: The average tree height of the training instances is 24, and some neighboring niveaus contain ~500,000 nodes each, summed up over all instances (although not all are selected thus evaluated). At tree height 20, we have already 99% of nodes. The biggest tree height with a selected node on top is 455. Tree height above 256 is but very rare, and should not lead to unexpected behaviour.
