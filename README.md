Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

The original version leads to a positive rate of ca. 94% and negative rate of ca. 87% in validation, mostly independent of the inspected hyperparameters, which can be explained as follows:

The cause is the compression algorithm and it's default setting: Before compression, for every problem, every clause gets assigned either a positive (+) or negative (-) weight of 1/num_clauses. Larger problems tend to have much larger sum of -/ sum of + ratios. By compression, smaller problems are packed into batches together with larger problems and the -/+ ratio is recomputed for the batch. 
Now, the loss function for the gradient descent algorithm is a BCE with logits loss and uses the -/+ ratio in a batch to equalize the class inbalance - leading to massive preference of positives for small problems: The average -/+ ratio of the 20422 problems is 70, minimum 0 and maximum 1645. 12502 problems are below ratio 70, 7920 problems above ratio 70.

One might argue, that for small problems, preference of positive rating is o.k., since they are solved easily, since only few initial axioms are present, and the other heuristics also do their job.

But from fixed embedding vectors, positively rated clauses with same derivation history will be rated positively in every problem because of this. Thereby, in large problems (or other small/medium-sized problems), such combinations will always be rated positively as well, which might be wrong.

Hence I modified the strategy for revealed axioms: Instead of a single embedding vector for all problems, they get assigned a vector v_i and a matrix A_i. Then, for every problem P, the embedding vectors concerning P are summed up, and axiom i is assigned the embedding vector A_i * sum_j(P) v_j.
