Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

The original version leads to a positive rate of ca. 94% and negative rate of ca. 87% in validation, mostly independent of the inspected hyperparameters.

Initials and derivations can be rated positive for some problems and negative for others. A fixed embedding vector can't classify problem-depending.

I modified the strategy for revealed axioms: Instead of a single embedding vector for all problems, they get assigned a vector v_i and a matrix A_i. Then, for every problem P, the embedding vectors concerning P are summed up, and axiom i is assigned the embedding vector A_i * sum_j(P) v_j.

I observe significantly improved loss and negative rate, but at the cost of positive rate.
