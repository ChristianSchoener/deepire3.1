Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

I modified the strategy for revealed axioms: Instead of a single embedding vector for all problems, they get assigned a vector v_i and a matrix A_i. Then, for every problem P, the embedding vectors concerning P are summed up, and axiom i is assigned A_i * sum_P v_j.

This modification allows axioms, which are classified positive as well as negative, to be correctly modeleld in every training problem. 
This is not possible with the single embedding vector of the original version.

First tests exhibit significantly improved loss and negative prediction, and also, they depend on embedding dimension. (One can imagine, if for example 500 axioms are revealed, and we have embedding dim 512, then we could choose as embedding vectors the standard unit vectors, whereby every axiom gets assigned a sum of some of its matrix columns. And as no pair of problems has identical axioms, revealed axioms can be resolved perfectly.)  

The avaluation of the results is ongoing.
