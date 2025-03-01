Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Remark (Research ongoing, but now slowly converging):

- To save training effort, a vectorization following a greedy approach was implemented, that bundles calculations of individual inference rules based on which can perform the most at a certain instant.

- Rule 52 (unit resulting resolution) does not lead to a selected node in any training instance used for testing (20,000 from unguided Vampire run with 10s time limit). And it's awfully slow to compute. It's banned for now, i.e., if new training instances arrive that contain a selected node which is derived by rule 52, it will still not be computed.

- The guidance helps to prove theorems if enough clauses are correctly rated negatively to prevent combinatoric explosion, while not making a single mistake for the positive ratings - clauses required for proofs are believed to be almost never replaceable. When all problems are merged and positive rating is preferd over negative, when there is both positive and negative, the resulting ratio is 1:50 pos/neg - a very difficult problem setting! To shed insight on how generalization works, several "Zero"-Tests with an analytical, instead of a statistically fitted behavior are being performed.
  1) Lookup clauses from solved problems (training set) and put positive / negative based on this experience. For clauses not seen in training, put positive. Result: Vampire with it's standard strategy, without guidance: 20632/57896 problems solved, and with the "zero" guidance, 20483/57896 problems. Stunningly, < 1% relative deviation. This shows, that the negatively rated clauses from the training set either interfere with other problems, or aren't enough to prevent combinatoric explosion. Since 90% of the negatively rated clauses are also negatively rated in the neural model, where 25% more problems are solved (instead of <1%), we conclude that the governing argument is that they aren't enough to prevent combinatoric explosion. (The standard heuristic is mixed in and picks among yet unevaluated clauses which is a bit different to what happens if no guidance is used. And it is not clear to me how well the ressources are utilized, i.e., if some of the parallel processes might receive less computational power - for the "Zero"-Test.)
  2) In the making: 1) Suggests, that the "unknown" axiom approach by Martin Suda has good generalization properties in a neighborhood of 25%-50% of the training data. Thereby we mean, that the rating of derivations from unseen axioms in about 25%-50% further problems is statistically resembled by derivations of less common axioms present in the training set and blanked out for this purpose. Therefore, we will do another "Zero"-Test: Consider the full merged multi-tree of all training data. For every problem, set those axioms to 0, which are not present in the specific problem, and also draw together the corresponding derivations. Guide the derivations of the axioms that are present in the problem but not in the training data by the former and corresponding derivations. We need to add a rule for derivations which are still unseen to obtain an algorithm. TODO (Maybe just positive in the first instance again?) The 50% additionally solves problems come from some experiments we did with revealing axioms and having them share an embedding vector, if they do not occur in the same problem. These were initial tests without much care, unfortunately, we don't have the complete data for it anymore. But I guess the insight provided by 1) and 2) is more interesting than random experiments.

If 2) above shows good generalization, we will argue that the generalization of Martin Suda's architecture is closely related to the fact that in the regime of the given training data, many small-medium sized problems have derivations that can be observed by the "unknown" axiom method at not too high tree heights and not too big distance from known derivations. Hence it is often times the case that unseen derivations can statistically be resembled from those originating from other axioms. For example a binary rule applied to axioms 0 and 0 is rated positively - this makes sense since for (almost) every problem, two of the unseen axioms yield a clause which is required for the proof. We know just of this compelling positive example yet and need to look up a few understandable negative ones. For problems not solved by the guidance, most certainly many more correct negative ratings are required to prevent combinatoric explosion. From my current understanding, the models the architecture yields do not generalize well enough to completely uncharted terrain - which is a very difficult problem because of the ratio 1:50 pos/neg.

Summing up - we will probably strengthen the oppinion, that the logic-agnostic approach requires more training data as compared to logic-aware approaches. (But I am waiting for a big surprise!)
