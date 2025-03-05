Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Remark (Close to finishing):

- To save training effort, a vectorization following a greedy approach was implemented, that bundles calculations of individual inference rules based on which can perform the most at a certain instant.

- Rule 52 (unit resulting resolution) does not lead to a selected node in any training instance used for testing (20,000 from unguided Vampire run with 10s time limit). And it's awfully slow to compute. It's banned for now, i.e., if new training instances arrive that contain a selected node which is derived by rule 52, it will still not be computed.

- The guidance helps to prove theorems if enough clauses are correctly rated negatively to prevent combinatoric explosion, while not making a single mistake for the positive ratings - clauses required for proofs are believed to be almost never replaceable. When all problems are merged and positive rating is preferd over negative, when there is both positive and negative, the resulting ratio is 1:50 pos/neg - a very difficult problem setting! 

- To shed insight on how generalization works, we compare the positive and negative derivations obtained without guidance, and with guidance by such model. (Work in progress)  
