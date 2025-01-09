Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

I modified the strategy for revealed axioms: They can have an overlapping ID, if they do not occur simultaneously in any training problem (and have same sine if it is used).
Thereby, ID=0 (unkwown axioms) doesn't receive much training, but we require an embedding for unseen axioms in Vampire later on.
Therefore, SWAPOUT is used.

To achieve all of this, a mapping from original axiom id to assigned axiom id is introduced in log_loader.py, then split up in the pieces in compressor.py, in the parallel training routine multi_inf_parallel_files_continuous.py, and employed in the recursive neural networks in inf_common.py. It is also exported in exporter.py.

TODO:
Furthermore is planned to be able to select between the current derivation embedding and a tree-LSTM embedding.
