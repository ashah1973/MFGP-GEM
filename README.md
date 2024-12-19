# MFGP-GEM
Multi-Fidelity Gaussian Process Model based on Graph Embeddings for Molecules.

This repository provides python codes and data for the paper 

"Rapid high-fidelity quantum simulations using multi-step nonlinear autoregression and graph embeddings", by A.A. Shah, P.K. Leung, W.W. Xing, to appear in npj Computational Materials

The uploaded code implements the method for the Alexandria data set, which is also uploaded (see the paper). I will add other examples and data sets, 
machine learning/multi-fidelity comparison codes and n-fidelity examples later.

The name of the code is 'MFGPGEM_Alexandria_G2.py'. In this example, the embeddings of the molecules and low-dimensional representations of additional descriptors are loaded from external files. To generate the latter, use the code 'Alexandria_G2_generate_Embediing_and_dimensionReduction.py'. 

For your own data, first use Alexandria_G2_generate_Embediing_and_dimensionReduction.py by supplying a SMILEs list in the code. The embeddings and low-dimensional representations will be saved. Then modify the file MFGPGEM_Alexandria_G2.py accordingly to load these files.
