Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Modifications:

1) "Stacking" strategies to reveal more axioms, which share common embedding vector in case they're not in any common problem (Need to re-integrate from older commits)
2) Macro mode: Every axiom gets it's own embedding vector +SWAPOUT
3) Matrix mode: Assign every revealed axiom a vecotr and a matrix, and for individual problems, mutiply the matrix with the sum of embedding vectors of the problem to obtain problem-dependent embeddings.
TODO: 4) Micro mode: For given problem, select a few training sets with corresponding axioms, train a mini-model and use it for guidance, repeat a few times
