Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Remark (Close to finishing):

- To save training effort, a vectorization following a greedy approach was implemented, that bundles calculations of individual inference rules based on which can perform the most at a certain instant.

- Rule 52 (unit resulting resolution) does not lead to a selected node in any training instance used for testing (20,000 from unguided Vampire run with 10s time limit). And it's awfully slow to compute. It's banned for now, i.e., if new training instances arrive that contain a selected node which is derived by rule 52, it will still not be computed.

- The guidance helps to prove theorems if enough clauses are correctly rated negatively to prevent combinatoric explosion, while not making a single mistake for the positive ratings - clauses required for proofs are believed to be almost never replaceable. When all problems are merged and positive rating is preferd over negative, when there is both positive and negative, the resulting ratio is 1:50 pos/neg - a very difficult problem setting! To shed insight on how generalization works, a "Zero"-Test with an analytical, instead of a statistically fitted behavior is being performed.

- In more detail: Lookup clauses from solved problems (training set) and put positive / negative based on this experience. For clauses not seen in training, put positive. Result: Vampire with it's standard strategy, without guidance: 20632/57896 problems solved, and with the "zero" guidance, 20483/57896 problems. Stunningly, < 1% relative deviation. This shows, that the negatively rated clauses from the training set either interfere with other problems, or aren't enough to prevent combinatoric explosion. Since 90% of the negatively rated clauses are also negatively rated in the neural model, where 25% more problems are solved (instead of <1%), we conclude that the governing argument is that they aren't enough to prevent combinatoric explosion. (The standard heuristic is mixed in and picks among yet unevaluated clauses which is a bit different to what happens if no guidance is used. And it is not clear to me how well the ressources are utilized, i.e., if some of the parallel processes might receive less computational ressources - for the "Zero"-Test. Update: 57592 problems took < 1 s neural model evaluation, 208 cases took between 1 and 2 seconds. So computation ressoures seem fine, i.e., there weren't a lot of hugh models or so.)

- We argue that the generalization of the model from Martin Suda's architecture is closely related to the fact that in the regime of the given training data, many small-medium sized problems have derivations similar in rating to those originating from "unknown" axioms at not too high tree heights and not too big distance from known derivations. For example a binary rule applied to axioms 0 and 0 is rated positively - this makes sense since for (almost) every problem, two of the unseen axioms yield a clause which is required for the proof. For problems not solved by the guidance, most certainly many more correct negative ratings are required to prevent combinatoric explosion. From our current understanding, the models have quite limited generalization to completely uncharted terrain - which is a very difficult problem, especially logic-agnostic, because of the ratio 1:50 pos/neg.

- Wanted to do more analytical tests, but it turns out they are super expensive to do.
