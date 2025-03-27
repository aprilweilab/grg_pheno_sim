"""
This file simulates binary phenotypes on GRGs by using the usual simulation methods
and then converting continuous phenotypes to binary phenotypes.
=======
"""

import pandas as pd
import scipy.stats as stats

from grg_pheno_sim.effect_size import (
    sim_grg_causal_mutation,
    additive_effect_sizes,
    samples_to_individuals,
    normalize_genetic_values,
    convert_to_effect_output,
)
from grg_pheno_sim.noise_sim import sim_env_noise
from grg_pheno_sim.normalization import normalize
from grg_pheno_sim.phenotype import convert_to_phen


def sim_binary_phenotypes(
    grg,
    model,
    num_causal,
    population_prevalence,
    random_seed,
    normalize_genetic_values_before_noise=False,
    noise_heritability=None,
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
    Since the function simulates binary phenotypes, we add a Gaussian threshold check
    at the very end to convert continuous values to binary values.

    Parameters
    ----------
    grg: The GRG on which phenotypes will be simulated.
    model: The distribution model from which effect sizes are drawn. Depends on the
    user's discretion.
    num_causal: Number of causal sites simulated.
    population_prevalence: The prevalence of the condition in the general population.
    0.1 means 1 in 10 individuals have the condition.
    random_seed: The random seed used for causal mutation effect simulation.
    normalize_genetic_values_before_noise: Checks whether to normalize the genetic
    values prior to simulating environmental noise (True if yes). Depends on the
    user's discretion. Set to False by default.
    noise_heritability: Takes in the h2 features to simulate environmental noise
    (set to None if the user prefers user-defined noise) and 1 is the user wants
    zero noise.
    user_mean: Mean parameter used for simulating environmental
    noise taken in from the user.
    user_cov: Covariance parameter used for simulating environmental
    noise taken in from the user.
    normalize_genetic_values_after: In the case where the h2 feature is not used,
    this checks whether the user wants genetic values normalized at the end (True
    if yes). Set to False by default.
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
    Pandas dataframe with resultant binary phenotypes. The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    """

    causal_mutation_df = sim_grg_causal_mutation(
        grg, num_causal=num_causal, model=model, random_seed=random_seed
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

    if noise_heritability is not None:
        phenotypes = sim_env_noise(individual_genetic_values, h2=noise_heritability)
        final_phenotypes = normalize(phenotypes)

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

        final_phenotypes = normalize(
            phenotypes, normalize_genetic_values=normalize_genetic_values_after
        )

    continuous_phen = final_phenotypes["phenotype"]
    k = population_prevalence
    T = stats.norm.ppf(1 - k)  # Gaussian threshold

    binary_phen = 1 * (continuous_phen >= T)

    final_phenotypes["phenotype"] = binary_phen

    if standardized_output == True:
        convert_to_phen(final_phenotypes, path, include_header=header)

    return final_phenotypes


def sim_binary_phenotypes_custom(
    grg,
    input_effects,
    population_prevalence,
    normalize_genetic_values_before_noise=False,
    noise_heritability=None,
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
    Since the function simulates binary phenotypes, we add a Gaussian threshold check
    at the very end to convert continuous values to binary values.

    Parameters
    ----------
    grg: The GRG on which phenotypes will be simulated.
    input_effects: The custom effect sizes dataset.
    population_prevalence: The prevalence of the condition in the general population.
    0.1 means 1 in 10 individuals have the condition.
    normalize_genetic_values_before_noise: Checks whether to normalize the genetic
    values prior to simulating environmental noise (True if yes). Depends on the
    user's discretion. Set to False by default.
    noise_heritability: Takes in the h2 features to simulate environmental noise
    (set to None if the user prefers user-defined noise) and 1 is the user wants
    zero noise.
    user_defined_noise_parameters: Parameters used for simulating environmental
    noise taken in from the user.
    normalize_genetic_values_after: In the case where the h2 feature is not used,
    this checks whether the user wants genetic values normalized at the end (True
    if yes). Set to False by default.
    save_effect_output: This boolean parameter decides whether the effect sizes
    will be saved to a .par file using the standard output format. Default value is False.
    effect_path: This parameter contains the path at which the .par output file
    will be saved. Default value is None.
    standardized_output: This boolean parameter decides whether the phenotypes
    will be saved to a .phen file using the standard output format. Default value is False.
    path: This parameter contains the path at which the .phen output file will be saved.
    Default value is None.
    header: This boolean parameter decides whether the .phen output file contains column
    headers or not. Default value is False.

    Returns
    --------------------
    Pandas dataframe with resultant binary phenotypes. The dataframe contains the following:
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

    if noise_heritability is not None:
        phenotypes = sim_env_noise(individual_genetic_values, h2=noise_heritability)
        final_phenotypes = normalize(phenotypes)

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

        final_phenotypes = normalize(
            phenotypes, normalize_genetic_values=normalize_genetic_values_after
        )

    continuous_phen = final_phenotypes["phenotype"]
    k = population_prevalence
    T = stats.norm.ppf(1 - k)  # Gaussian threshold

    binary_phen = 1 * (continuous_phen >= T)

    final_phenotypes["phenotype"] = binary_phen

    if standardized_output == True:

        convert_to_phen(final_phenotypes, path, include_header=header)

    return final_phenotypes
