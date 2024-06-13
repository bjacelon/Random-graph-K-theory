This directory contains four Python scripts for the generation and analysis of random graphs, accompanying the paper
  The K-distribution of random graph C^*-algebras
    by Bhishan Jacelon, Igor Khavkine
  https://arxiv.org/abs/2307.01861

1. dnq.py: generates random Bernoulli digraphs D_{n,q} (or shifted digraphs D_{n,q}+I) and collects K-theoretic, automorphism orbit and determinandt data.
2. enq.py: generates random Erdos-Renyi graphs E_{n,q} and collects K-theoretic, automorphism orbit and determinandt data.
3. gnr.py:  generates random r-regular multigraphs G_{n,r} and collects K-theoretic, automorphism orbit and determinandt data.
4. randomgraphktheory.py: collects information about the Sylow subgroups of the K_0 groups for the three above graph models.

These can be executed for example with the following options:

* python3 dnq.py 100 1/2 -m 10000
  
    generates 10000 random graphs D_{100,1/2} and outputs the data as a csv file
  
* python3 enq.py 100 1/2 -m 10000
  
    generates 10000 random graphs E_{100,1/2} and outputs the data as a csv file
  
* python3 gnr.py 100 17 -m 10000
  
    generates 10000 random graphs G_{100,17} and outputs the data as a csv file
  
* python3 randomgraphktheory.py
  
    interactively asks for the graph model and outputs the data as a csv file

Running the commands with no options will either ask for interactive input or ask for arguments.

Nontrivial Python module dependencies (installable with pip):
  networkx, matplotlib, numpy, scipy, sympy, cypari

Also included in the directory are summary experimental data as csv files for the various random graph models considered in the paper.
