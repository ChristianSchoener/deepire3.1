Derived from supplementary material (see https://github.com/quickbeam123/deepire3.1) to 

https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11

Remark:

After trying out several things to save computation time for the training, I have come to the idea that a different setup of the neural network might be beneficial:

Modeling positive rated vectors as non-zero vectors, negative rated clauses as the zero vector; therefore removing biases from layers, making activation into tanh, and turning concatenation into component-wise product or convolution.

Thereby, it suffices to keep the positive "cone" leading to proofs, and attach 1 layer of negative boundary, but not add so much negative that the selected values show up (since zero vectors will stay zero vectors by construction of the neural network). This yields just about ~350,000 nodes instead of ~10,000,000 and can be trained very quickly.

It is unclear to me, which approach models "unseen" situations better. This new approach has the benefit, that negative stays negative, and cannot be turned positive by some circumstances.

I have the hope, that the approach will lead to unseen derivations being rated positively more often, since producing the (close to) zero vector should be difficult?

If clauses are rated negatively, they will maybe never be activated. But if they are required, the proof is impossible due to the guidance.

Rating positive more aggressive is hence better.

Let's see! :)
