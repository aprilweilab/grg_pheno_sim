"""
This file simulates phenotypes on multiple GRGs by using the usual simulation methods.
=======
"""

import numpy as np
import pandas as pd
import pygrgl
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


def intermediate_genetic_vals(
    grg,
    model,
    num_causal,
    random_seed,
    normalize_genetic_values_before_noise,
    grg_name,
    effect_path,
):
    """
    Intermediate function to simulate effect sizes and compute genetic values
    for each individual GRG.
    """
    causal_mutation_df = sim_grg_causal_mutation(
        grg, num_causal=num_causal, model=model, random_seed=random_seed
    )

    if effect_path is not None:
        convert_to_effect_output(causal_mutation_df, grg, effect_path)

    genetic_values = additive_effect_sizes(grg, causal_mutation_df)

    individual_genetic_values = samples_to_individuals(genetic_values)

    if normalize_genetic_values_before_noise == True:
        individual_genetic_values = normalize_genetic_values(individual_genetic_values)

    print("Genetic values for " + grg_name + " are as follows:")
    print(individual_genetic_values)

    return individual_genetic_values


def sim_phenotypes_multi_grg_ram(
    grg_files,
    model,
    num_causal_per_file,
    random_seed,
    normalize_genetic_values_before_noise,
    population_prev,
    noise_heritability,
    user_mean,
    user_cov,
    normalize_genetic_values_after,
    save_effect_output,
    effect_path_list,
    standardized_output,
    path,
    header,
):
    """
    Simulate phenotypes by loading all GRGs into RAM simultaneously.

    Parameters
    ----------
    grg_files : List of paths to GRG files to be processed.
    model: The distribution model from which effect sizes are drawn. Depends on the user's discretion.
    num_causal_per_file: Number of causal sites simulated for each file (same for each GRG).
    random_seed: The random seed used for causal mutation simulation.
    normalize_genetic_values_before_noise: Checks whether to normalize the genetic values prior to simulating environmental noise (True if yes). Depends on the user's discretion. Set to False by default.
    noise_heritability: Takes in the h2 features to simulate environmental noise (set to None if the user prefers user-defined noise) and 1 is the user wants zero noise.
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
    """
    # Load all GRGs
    all_grgs = []
    for grg_file in grg_files:
        temp_grg = pygrgl.load_immutable_grg(grg_file)
        all_grgs.append(temp_grg)
        print("Loaded " + str(grg_file) + " into RAM")

    # Collect genetic value dataframes
    all_genetic_values = []
    index = 0
    for grg in all_grgs:

        if save_effect_output:
            path = effect_path_list[index]
        else:
            path = None

        grg_name = grg_files[index]
        genetic_val_df = intermediate_genetic_vals(
            grg,
            model,
            num_causal_per_file,
            np.random.seed(random_seed),
            normalize_genetic_values_before_noise,
            grg_name,
            effect_path=path,
        )
        all_genetic_values.append(genetic_val_df)
        index += 1

    combined_genetic_df = (
        pd.concat(all_genetic_values)
        .groupby(["individual_id", "causal_mutation_id"], as_index=False)
        .agg({"genetic_value": "sum"})
    )

    combined_genetic_df = combined_genetic_df[
        ["individual_id", "genetic_value", "causal_mutation_id"]
    ]

    print("The combined genetic values data is given below ")
    print(combined_genetic_df)

    causal_mutation_id = combined_genetic_df["causal_mutation_id"].unique()
    check = len(causal_mutation_id) == 1

    if noise_heritability is not None:
        phenotypes = sim_env_noise(combined_genetic_df, h2=noise_heritability)
        final_phenotypes = normalize(phenotypes)

    else:
        if check:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, mean=user_mean, std=user_cov
            )
        else:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, means=user_mean, cov=user_cov
            )

        final_phenotypes = normalize(
            phenotypes, normalize_genetic_values=normalize_genetic_values_after
        )

    if population_prev is not None:
        continuous_phen = final_phenotypes["phenotype"]
        k = population_prev
        T = stats.norm.ppf(1 - k)  # Gaussian threshold

        binary_phen = 1 * (continuous_phen >= T)

        final_phenotypes["phenotype"] = binary_phen

    if standardized_output == True:
        convert_to_phen(final_phenotypes, path, include_header=header)

    return final_phenotypes


def sim_phenotypes_multi_grg_sequential(
    grg_files,
    model,
    num_causal_per_file,
    random_seed,
    normalize_genetic_values_before_noise,
    population_prev,
    noise_heritability,
    user_mean,
    user_cov,
    normalize_genetic_values_after,
    save_effect_output,
    effect_path_list,
    standardized_output,
    path,
    header,
):
    """
    Simulate phenotypes by processing GRGs sequentially to reduce memory usage.

    Parameters
    ----------
    grg_files : List of paths to GRG files to be processed
    model: The distribution model from which effect sizes are drawn. Depends on the user's discretion.
    num_causal_per_file: Number of causal sites simulated for each file (same for each GRG).
    random_seed: The random seed used for causal mutation simulation.
    normalize_genetic_values_before_noise: Checks whether to normalize the genetic values prior to simulating environmental noise (True if yes). Depends on the user's discretion. Set to False by default.
    noise_heritability: Takes in the h2 features to simulate environmental noise (set to None if the user prefers user-defined noise) and 1 is the user wants zero noise.
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
    """

    all_genetic_values = []

    # Load GRGs sequentially
    index = 0
    for grg_file in grg_files:
        temp_grg = pygrgl.load_immutable_grg(grg_file)
        print("Loaded " + str(grg_file) + " into RAM")

        if save_effect_output:
            path = effect_path_list[index]
        else:
            path = None

        genetic_val_df = intermediate_genetic_vals(
            temp_grg,
            model,
            num_causal_per_file,
            np.random.seed(random_seed),
            normalize_genetic_values_before_noise,
            grg_file,
            effect_path=path,
        )
        all_genetic_values.append(genetic_val_df)

        index += 1

    combined_genetic_df = (
        pd.concat(all_genetic_values)
        .groupby(["individual_id", "causal_mutation_id"], as_index=False)
        .agg({"genetic_value": "sum"})
    )

    combined_genetic_df = combined_genetic_df[
        ["individual_id", "genetic_value", "causal_mutation_id"]
    ]

    print("The combined genetic values data is given below ")
    print(combined_genetic_df)

    causal_mutation_id = combined_genetic_df["causal_mutation_id"].unique()
    check = len(causal_mutation_id) == 1

    if noise_heritability is not None:
        phenotypes = sim_env_noise(combined_genetic_df, h2=noise_heritability)
        final_phenotypes = normalize(phenotypes)

    else:
        if check:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, mean=user_mean, std=user_cov
            )
        else:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, means=user_mean, cov=user_cov
            )

        final_phenotypes = normalize(
            phenotypes, normalize_genetic_values=normalize_genetic_values_after
        )

    if population_prev is not None:
        continuous_phen = final_phenotypes["phenotype"]
        k = population_prev
        T = stats.norm.ppf(1 - k)  # Gaussian threshold

        binary_phen = 1 * (continuous_phen >= T)

        final_phenotypes["phenotype"] = binary_phen

    if standardized_output == True:
        convert_to_phen(final_phenotypes, path, include_header=header)

    return final_phenotypes


def sim_phenotypes_multi_grg(
    grg_files,
    model,
    num_causal_per_file,
    random_seed,
    load_all_ram=False,
    normalize_genetic_values_before_noise=False,
    noise_heritability=None,
    binary=False,
    population_prevalence=None,
    user_mean=None,
    user_cov=None,
    normalize_genetic_values_after=False,
    save_effect_output=False,
    effect_path_list=None,
    standardized_output=False,
    path=None,
    header=False,
):
    """
    Simulate phenotypes across multiple GRG files with two loading strategies.

    Parameters
    ----------
    grg_files : list
        List of paths to GRG files to be processed
    load_all_ram : bool, optional
        If True, load all GRGs into RAM and sample causal variants simultaneously.
        If False, process GRGs sequentially to reduce memory usage.

    Other parameters are similar to sim_phenotypes function:
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
    # Validate inputs
    if not isinstance(grg_files, list):
        raise ValueError("grg_files must be a list of file paths")

    if binary:
        population_prev = population_prevalence
    else:
        population_prev = None

    if load_all_ram:
        # Strategy 1: Load all GRGs into RAM
        return sim_phenotypes_multi_grg_ram(
            grg_files,
            model,
            num_causal_per_file,
            random_seed,
            normalize_genetic_values_before_noise,
            population_prev,
            noise_heritability,
            user_mean,
            user_cov,
            normalize_genetic_values_after,
            save_effect_output,
            effect_path_list,
            standardized_output,
            path,
            header,
        )
    else:
        # Strategy 2: Process GRGs sequentially
        return sim_phenotypes_multi_grg_sequential(
            grg_files,
            model,
            num_causal_per_file,
            random_seed,
            normalize_genetic_values_before_noise,
            population_prev,
            noise_heritability,
            user_mean,
            user_cov,
            normalize_genetic_values_after,
            save_effect_output,
            effect_path_list,
            standardized_output,
            path,
            header,
        )
