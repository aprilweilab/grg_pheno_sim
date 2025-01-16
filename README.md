# grg_pheno_sim
This is a code repository to simulates phenotypes on GRGs (genotype representation graphs). The simulator first simulates effect sizes based on the user's desired distribution model (a wide spectrum of options are provided, both for simulation of single and multiple causal mutations at a go), computes the genetic values by passing the effect sizes down the genotype representation graph, and then adds simulated environmental noise to obtain the final phenotypes for the individuals in the graph. Normalization of genetic values is provided as well, either prior to adding environmental noise or after noise is added, according to the user's desire. The simulator offers the simulation of binary phenotypes as well, in addition to simulation on multiple GRGs simultaneously. Finally, options to obtain standardized outputs for both effect sizes (.par files) and phenotypes (.phen files) are included as well.

The folder `grg_pheno_sim` contains all the primary source code for the simulator. The `demos` folder contains ipynb notebooks with sample uses and demomstrations of the different stages of the phenotype simulator. It also contains incremental verifications of outputs to ensure accurate simulation. The `test_phenotype_sim` folder contains a suite of test functions used in the demos.

# Installation

_Installing from pip_ 

If you just want to use the tools offered by `grg_pheno_sim` then you can install via pip (from [PyPi](http://pypi.org/project/grg_pheno_sim/))

`pip install grg_pheno_sim`

_Installing from source_

1) Clone the repository
2) If you wish to install the package without any changes to source code, use `pip install /path/to/grg_pheno_sim/` (this is for standard installation)
3) If you wish to install the package and modify the source code, use `pip install -e /path/to/grg_pheno_sim/` (this is for development installation)

# Usage

The `demos` folder contains a vast repository of use cases for the phenotype simulator, including sample outputs and standardized outputs commands (the output files themselves are excluded from the GitHub repo but can be easily obtained by running the appropriate notebook).


