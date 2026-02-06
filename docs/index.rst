grg_pheno_sim Documentation
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   API Reference <grg_pheno_sim>

``grg_pheno_sim`` simulates phenotypes on `GRGs (Genotype Representation Graphs) <https://grgl.readthedocs.io/en/stable/concepts.html>`_.
The simulator first simulates effect sizes based on the user's desired distribution model (a wide spectrum of options are provided, both
for simulation of single and multiple causal mutations at a go), computes the genetic values by passing the effect sizes down the GRG,
and then adds simulated environmental noise to obtain the final phenotypes for the individuals in the graph. Normalization of genetic
values is provided as well, either prior to adding environmental noise or after noise is added, according to the user's desire. In
addition, there is an option to use normalized genotypes. The simulator offers the simulation of binary phenotypes as well, in addition
to simulation on multiple GRGs simultaneously. Finally, options to obtain standardized outputs for both effect sizes (.par files) and
phenotypes (.phen files) are included as well.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
