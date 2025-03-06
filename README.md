Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Remark (Close to finishing):

- To save training effort, a vectorization following a greedy approach was implemented, that bundles calculations of individual inference rules based on which can perform the most at a certain instant.

- The guidance helps to prove theorems if enough clauses are correctly rated negatively to prevent combinatoric explosion, while making avery few mistakes for the positive ratings - such clauses will be looked at at some point by the usual heuristic, but this situation might not be recoverable because of the time limit. When all problems are merged and positive rating is prefered over negative rating in cases where there is both positive and negative in some problems, the resulting ratio is 1:50 pos/neg - a very difficult problem setting! 

- To shed insight into how generalization works, we compare the positive and negative derivations obtained without guidance, and those obtained with guidance by such model.
