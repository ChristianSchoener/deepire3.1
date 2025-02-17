Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Modifications:

1) Vectorization of the evaluation scheme by a greedy approach: Instead of performing computations one after another, the derivation rule for which the most derivations can be made simultaneously is chosen. Thereby, the number of function calls is reduced dramatically, by a factor of 10-100, and the computations are performed in an efficient vectorized manner.
2) All problems arereduced for unnecessary derivations, i.e., derivations which have no labeled nodes in their sub-tree.
3) Furthermore, all problems are scanned for common derivations, the weights are gathered into a single occurence, and then all other occurences are deleted, if above them is no node with a label. Afterwards, the gathered weights are spread uniformly over the remaining occurences of the derivation.

As an illustration: After loading the log files, the main data file is about ~2.5 GiB big. After step 2) about ~1.5 GiB, and after step 3), it is ~0.55 GiB.

Furthermore, I implemented what is necessary to use GPU for the training computations.

On my PC, with 500 revealed axioms, training time decayed from ~24 minutes/epoch to <5 minutes/epoch. 
