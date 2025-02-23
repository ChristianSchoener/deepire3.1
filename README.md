Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Remark (Research ongoing):

- Evaluation now has a scalar product for every tree height of derivations instead of just a single one, making evaluation more fine-grained and thus capable. (In fact, just heights 1-255 cyclic)  

As a background info: The average tree height of the training instances is 24, and some neighboring niveaus contain ~500,000 nodes each, summed up over all instances (although not all are selected thus evaluated). At tree height 20, we have already 99% of nodes. The biggest tree height with a selected node on top is 455. Tree height above 256 is but very rare, and should not lead to unexpected behaviour.

 Evaluation is on-going, but preliminary tests exhibit good performance of Vampire guided with such model.

- To save training effort, a vectorization following a greedy approach was implemented, that bundles calculations of individual inference rules based on which can perform the most in an instant.

- Rule 52 (unit resulting resolution) does not lead to a selected node in any training instance used for testing (20,000 from unguided Vampire run with 10s time limit). And it's awfully slow to compute. It's banned for now, i.e., if new training instances arrive that contain a selected node which is derived by rule 52, it will still not be computed.

- More effort has been spent on the fine-tuning of the preprocessing to just keep the (multi-)trees generated by the selections, removing all values above which do not lead to a selected node. This saves quite some computation time, as PyTorch doesn't seem to take notice for backpropagation, that this part of the graph can just be deleted.
