"""
This file simulates the phenotypes overall by combining the incremental stages of simulation on GRGs.
=======
"""

import pandas as pd
import numpy as np
import pygrgl

from grg_pheno_sim.effect_size import (
    sim_grg_causal_mutation,
    additive_effect_sizes,
    samples_to_individuals,
    normalize_genetic_values,
    convert_to_effect_output,
)
from grg_pheno_sim.noise_sim import sim_env_noise
from grg_pheno_sim.model import grg_causal_mutation_model
from grg_pheno_sim.normalization import normalize
from grg_pheno_sim.effect_size import allele_frequencies
from grg_pheno_sim.ops_scipy import SciPyStdXOperator as _SciPyStdXOperator

def phenotype_class_to_df(phenotypes):
    """This function performs extracts the dataframe and performs
    necessary modifications before returning it.
    """
    dataframe = phenotypes.get_df()
    dataframe["individual_id"] = dataframe["individual_id"].astype(int)
    dataframe["causal_mutation_id"] = dataframe["causal_mutation_id"].astype(int)
    return dataframe


def convert_to_phen(phenotypes_df, path, include_header=False):
    """
    This function converts the phenotypes dataframe to a CSV file.

    Parameters
    ----------
    phenotypes_df: The input pandas dataframe containing the phenotypes.
    path: The path at which the CSV file will be saved.
    include_header: A boolean parameter that indicates whether headers have to be included.
    Default value is False.
    """
    if path is None:
        raise ValueError("Output path must be defined")

    df_phen = phenotypes_df[["individual_id", "phenotype"]].rename(
        columns={"individual_id": "person_id", "phenotype": "phenotypes"}
    )

    df_phen.to_csv(path, sep="\t", index=False, header=include_header)


def sim_phenotypes(
    grg,
    model=grg_causal_mutation_model("normal", mean=0, var=1),
    num_causal=1000,
    random_seed=42,
    normalize_phenotype=False,
    normalize_genetic_values_before_noise=False,
    heritability=None,
    user_mean=None,
    user_cov=None,
    normalize_genetic_values_after=False,
    save_effect_output=False,
    effect_path=None,
    standardized_output=False,
    path=None,
    header=False,
    standardized=False
):
    """
    Function to simulate phenotypes in one go by combining all intermittent stages.

    Parameters
    ----------
    grg: The GRG on which phenotypes will be simulated.
    model: The distribution model from which effect sizes are drawn. Depends on the user's discretion.
    Default model used is the standard Gaussian. 
    num_causal: Number of causal sites simulated. Default value used is 1000.
    random_seed: The random seed used for causal mutation simulation. Default values is 42.
    normalize_phenotype: Checks whether to normalize the phenotypes. The default value is False.
    normalize_genetic_values_before_noise: Checks whether to normalize the genetic values prior to simulating environmental noise (True if yes). Depends on the user's discretion. Set to False by default.
    heritability: Takes in the h2 features to simulate environmental noise (set to None if the user prefers user-defined noise) and 1 is the user wants zero noise.
    user_defined_noise_parameters: Parameters used for simulating environmental noise taken in from the user.
    normalize_genetic_values_after: In the case where the h2 feature is not used, this checks whether the user wants genetic values normalized at the end (True if yes). Set to False by default.
    save_effect_output: This boolean parameter decides whether the effect sizes
    will be saved to a .par file using the standard output format. Default value is False.
    effect_path: This parameter contains the path at which the .par output file will be saved.
    Default value is None.
    standardized_output: This boolean parameter decides whether the phenotypes
    will be saved to a .phen file using the standard output format. Default value is False.
    path: This parameter contains the path at which the .phen output file will be saved.
    Default value is None.
    header: This boolean parameter decides whether the .phen output file contains column
    headers or not. Default value is False.
    standardized: This boolean parameters decides whether the simulation uses standardized genotypes.

    Returns
    --------------------
    Pandas dataframe with resultant phenotypes. The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    """

    if standardized is True:
        return sim_phenotypes_standardized(grg, heritability, num_causal, random_seed, save_effect_output,
                                           effect_path, standardized_output, path, header)

    causal_mutation_df = sim_grg_causal_mutation(
        grg, heritability, num_causal=num_causal, model=model, random_seed=random_seed
    )

    print("The initial effect sizes are ")
    print(causal_mutation_df)

    if save_effect_output == True:
        convert_to_effect_output(causal_mutation_df, grg, effect_path)

    genetic_values = additive_effect_sizes(grg, causal_mutation_df)
    causal_mutation_id = genetic_values["causal_mutation_id"].unique()
    check = len(causal_mutation_id) == 1

    individual_genetic_values = samples_to_individuals(genetic_values)

    print("The genetic values of the individuals are ")
    print(individual_genetic_values)

    if normalize_genetic_values_before_noise == True:
        individual_genetic_values = normalize_genetic_values(individual_genetic_values)

    if heritability is not None:
        phenotypes = sim_env_noise(individual_genetic_values, h2=heritability)
        if normalize_phenotype:
            final_phenotypes = normalize(phenotypes)
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

    else:
        if check:
            phenotypes = sim_env_noise(
                individual_genetic_values,
                user_defined=True,
                mean=user_mean,
                std=user_cov,
            )
        else:
            phenotypes = sim_env_noise(
                individual_genetic_values,
                user_defined=True,
                means=user_mean,
                cov=user_cov,
            )

        if normalize_phenotype:
            final_phenotypes = normalize(
                phenotypes, normalize_genetic_values=normalize_genetic_values_after
            )
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

    if standardized_output == True:
        convert_to_phen(final_phenotypes, path, include_header=header)

    return final_phenotypes


def sim_phenotypes_custom(
    grg,
    input_effects,
    normalize_phenotype=False,
    normalize_genetic_values_before_noise=False,
    heritability=None,
    user_mean=None,
    user_cov=None,
    normalize_genetic_values_after=False,
    save_effect_output=False,
    effect_path=None,
    standardized_output=False,
    path=None,
    header=False,
):
    """
    Function to simulate phenotypes in one go by combining all intermittent stages.
    This function accepts custom effect sizes instead of simulating them using
    the causal mutation models.

    Parameters
    ----------
    grg: The GRG on which phenotypes will be simulated.
    input_effects: The custom effect sizes dataset.
    normalize_phenotype: Checks whether to normalize the phenotypes. The default value is False.
    normalize_genetic_values_before_noise: Checks whether to normalize the genetic values prior to simulating environmental noise (True if yes). Depends on the user's discretion. Set to False by default.
    heritability: Takes in the h2 features to simulate environmental noise (set to None if the user prefers user-defined noise) and 1 is the user wants zero noise.
    user_defined_noise_parameters: Parameters used for simulating environmental noise taken in from the user.
    normalize_genetic_values_after: In the case where the h2 feature is not used, this checks whether the user wants genetic values normalized at the end (True if yes). Set to False by default.
    save_effect_output: This boolean parameter decides whether the effect sizes
    will be saved to a .par file using the standard output format. Default value is False.
    effect_path: This parameter contains the path at which the .par output file will be saved.
    Default value is None.
    standardized_output: This boolean parameter decides whether the phenotypes
    will be saved to a .phen file using the standard output format. Default value is False.
    path: This parameter contains the path at which the .phen output file will be saved.
    Default value is None.
    header: This boolean parameter decides whether the .phen output file contains column
    headers or not. Default value is False.

    Returns
    --------------------
    Pandas dataframe with resultant phenotypes. The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    """

    if isinstance(input_effects, dict):
        causal_mutation_df = pd.DataFrame(
            list(input_effects.items()), columns=["mutation_id", "effect_size"]
        )
        causal_mutation_df["causal_mutation_id"] = 0
    elif isinstance(input_effects, list):
        causal_mutation_df = pd.DataFrame(input_effects, columns=["effect_size"])
        causal_mutation_df["mutation_id"] = causal_mutation_df.index
        causal_mutation_df = causal_mutation_df[["mutation_id", "effect_size"]]
        causal_mutation_df["causal_mutation_id"] = 0
    elif isinstance(input_effects, pd.DataFrame):
        causal_mutation_df = input_effects
        causal_mutation_df["causal_mutation_id"] = 0

    print("The initial effect sizes are ")
    print(causal_mutation_df)

    if save_effect_output == True:
        convert_to_effect_output(causal_mutation_df, grg, effect_path)

    genetic_values = additive_effect_sizes(grg, causal_mutation_df)
    causal_mutation_id = genetic_values["causal_mutation_id"].unique()
    check = len(causal_mutation_id) == 1

    individual_genetic_values = samples_to_individuals(genetic_values)

    print("The genetic values of the individuals are ")
    print(individual_genetic_values)

    if normalize_genetic_values_before_noise == True:
        individual_genetic_values = normalize_genetic_values(individual_genetic_values)

    if heritability is not None:
        phenotypes = sim_env_noise(individual_genetic_values, h2=heritability)
        if normalize_phenotype:
            final_phenotypes = normalize(phenotypes)
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

    else:
        if check:
            phenotypes = sim_env_noise(
                individual_genetic_values,
                user_defined=True,
                mean=user_mean,
                std=user_cov,
            )
        else:
            phenotypes = sim_env_noise(
                individual_genetic_values,
                user_defined=True,
                means=user_mean,
                cov=user_cov,
            )

        if normalize_phenotype:
            final_phenotypes = normalize(
                phenotypes, normalize_genetic_values=normalize_genetic_values_after
            )
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

    if standardized_output == True:

        convert_to_phen(final_phenotypes, path, include_header=header)

    return final_phenotypes

def sim_phenotypes_standardized(
    grg,
    heritability,
    num_causal,
    random_seed, 
    save_effect_output,
    effect_path, 
    standardized_output, 
    path, 
    header
):
    """
    Function to simulate phenotypes using standardized genotype matrices.
    
    Based on the standardized approach where X' = (X-U)Σ, and genetic values
    are computed as X'β = X(Σβ) - UΣβ.

    Parameters
    ----------
    grg: The GRG on which phenotypes will be simulated.
    heritability: The narrow sense heritability (h²).
    num_causal: Number of causal mutations to be simulated.
    random_seed: Random seed for reproducibility.

    Returns
    --------------------
    Pandas dataframe with resultant phenotypes. The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    """

    # Sample effect sizes from normal distribution with variance h²/M_causal
    mean_1 = 0.0  
    var_1 = heritability / num_causal
    model_normal = grg_causal_mutation_model("normal", mean=mean_1, var=var_1)

    # Simulate causal mutations and their effect sizes
    causal_mutation_df = sim_grg_causal_mutation(
        grg, model=model_normal, num_causal=num_causal, random_seed=random_seed
    )

    print("The initial effect sizes are ")
    print(causal_mutation_df)

    # Get causal mutation sites and their effect sizes
    causal_sites = causal_mutation_df["mutation_id"].values
    effect_sizes = causal_mutation_df["effect_size"].values
    # Calculate allele frequencies for causal sites
    frequencies = allele_frequencies(grg, causal_mutation_df)
    # Create standardized effect size vector Σβ
    # Σ is diagonal matrix with elements 1/σᵢ where σᵢ = √(2fᵢ(1-fᵢ))
    # So Σβ has elements βᵢ/σᵢ at each causal site
    num_mutations = grg.num_mutations
    standardized_effect_vector = np.zeros(num_mutations, dtype=np.float64)
    for i, (site, beta, freq) in enumerate(zip(causal_sites, effect_sizes, frequencies)):
        # Calculate standard deviation: σᵢ = √(2fᵢ(1-fᵢ))
        sigma_i = np.sqrt(2 * freq * (1 - freq))
        if sigma_i > 0:  # Avoid division by zero
            standardized_effect_vector[site] = beta / sigma_i
    # Calculate X(Σβ) using dot product with DOWN direction
    # This gives us the raw genetic values before allele frequency adjustment
    raw_genetic_values = pygrgl.dot_product(
        grg=grg, 
        input=standardized_effect_vector, 
        direction=pygrgl.TraversalDirection.DOWN
    )
    print("Raw genetic ", raw_genetic_values[0]+ raw_genetic_values[1])

    # Calculate UΣβ (allele frequency adjustment term)
    # U is N-by-M matrix where each entry in i-th column is 2fᵢ
    # So UΣβ is N-by-1 vector filled with Σᵢ 2fᵢβᵢ/σᵢ
    allele_freq_adjustment = 0.0
    for i, (site, beta, freq) in enumerate(zip(causal_sites, effect_sizes, frequencies)):
        sigma_i = np.sqrt(2 * freq * (1 - freq))
        if sigma_i > 0:
            allele_freq_adjustment += (2 * freq * beta) / sigma_i
    # Get sample nodes and calculate final genetic values: X'β = X(Σβ) - UΣβ
    samples_list = grg.get_sample_nodes()
    final_genetic_values = []
    
    for node in samples_list:
        genetic_value = raw_genetic_values[node] - 0.5 *allele_freq_adjustment
        final_genetic_values.append(genetic_value)
    # Create DataFrame with sample-level genetic values
    sample_effects_df = pd.DataFrame({
        "sample_node_id": samples_list,
        "genetic_value": final_genetic_values,
        "causal_mutation_id": 0
    })
    
    # Convert sample-level genetic values to individual-level
    individual_genetic_values = samples_to_individuals(sample_effects_df)

    print("The genetic values of the individuals are ")
    print(individual_genetic_values)

    # Calculate variance of genetic values for environmental noise simulation
    # Environmental noise: ε ~ N(0, Var(Xβ)(1/h² - 1))
    genetic_var = individual_genetic_values["genetic_value"].var()
    noise_var = genetic_var * (1/(heritability) - 1)
    
    # Simulate environmental noise
    rng = np.random.default_rng(random_seed)
    num_individuals = len(individual_genetic_values)
    environmental_noise = rng.normal(0, np.sqrt(noise_var), size=num_individuals)
    
    # Create final phenotype dataframe
    final_phenotypes = individual_genetic_values.copy()
    final_phenotypes["environmental_noise"] = environmental_noise
    final_phenotypes["phenotype"] = (
        final_phenotypes["genetic_value"] + final_phenotypes["environmental_noise"]
    )

    return final_phenotypes

def allele_frequencies_new(grg: pygrgl.GRG) -> np.typing.NDArray:
    """
    Get the allele frequencies for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: A vector of length grg.num_mutations, containing allele frequencies
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    return pygrgl.matmul(
        grg,
        np.ones((1, grg.num_samples), dtype=np.int32),
        pygrgl.TraversalDirection.UP,
    )[0] / (grg.num_samples)

def sim_phenotypes_StdOp(grg,
    heritability,
    num_causal=1000,
    random_seed = 42
):    
    # Sample effect sizes from normal distribution with variance h²/M_causal
    mean_1 = 0.0  
    var_1 = heritability / num_causal
    model_normal = grg_causal_mutation_model("normal", mean=mean_1, var=var_1)

    # Simulate causal mutations and their effect sizes
    causal_mutation_df = sim_grg_causal_mutation(
        grg, model=model_normal, num_causal=num_causal, random_seed=random_seed
    )
        # Get causal mutation sites and their effect sizes
    causal_sites = causal_mutation_df["mutation_id"].values
    effect_sizes = causal_mutation_df["effect_size"].values

    freqs = allele_frequencies_new(grg)  
    beta_full = np.zeros(grg.num_mutations, dtype=float)
    beta_full[causal_sites] = causal_mutation_df["effect_size"].values
    print(beta_full[48])
    beta_full = beta_full.reshape(-1,1)
    standard_gv = _SciPyStdXOperator(grg, direction= pygrgl.TraversalDirection.UP, freqs = freqs, haploid= False)._matmat(beta_full)
    for i in range(20):
        print(standard_gv[i][0])