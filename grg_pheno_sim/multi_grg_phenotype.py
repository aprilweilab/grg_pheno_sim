"""
This module simulates phenotypes on multiple GRGs by using the usual simulation methods.
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
from grg_pheno_sim.model import grg_causal_mutation_model
from grg_pheno_sim.normalization import normalize
from grg_pheno_sim.ops_scipy import SciPyStdXOperator as _SciPyStdXOperator
from grg_pheno_sim.phenotype import (
    allele_frequencies_new,
    convert_to_phen,
    phenotype_class_to_df,
)


def intermediate_genetic_vals(
    grg,
    model,
    num_causal,
    random_seed,
    normalize_genetic_values_before_noise,
    grg_name,
    effect_path,
    standardized,
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

    ###This also allows for multivariate causal mutations
    if standardized:
        freqs = allele_frequencies_new(grg)
        causal_ids = np.sort(causal_mutation_df["causal_mutation_id"].unique())
        beta = np.zeros((grg.num_mutations, len(causal_ids)), dtype=float)

        for beta_col, causal_id in enumerate(causal_ids):
            causal_subset = causal_mutation_df[
                causal_mutation_df["causal_mutation_id"] == causal_id
            ]
            mutation_ids = causal_subset["mutation_id"].astype(int).to_numpy()
            beta[mutation_ids, beta_col] = causal_subset["effect_size"].to_numpy()

        gv = _SciPyStdXOperator(
            grg,
            direction=pygrgl.TraversalDirection.UP,
            freqs=freqs,
            haploid=False,
        )._matmat(beta)

        if gv.ndim == 1:
            gv = gv.reshape(-1, 1)

        individual_genetic_values = pd.DataFrame(
            {
                "individual_id": np.repeat(
                    np.arange(grg.num_individuals), len(causal_ids)
                ),
                "genetic_value": gv.reshape(-1),
                "causal_mutation_id": np.tile(causal_ids, grg.num_individuals),
            }
        )
    else:
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
    normalize_phenotype,
    normalize_genetic_values_before_noise,
    population_prev,
    heritability,
    user_mean,
    user_cov,
    normalize_genetic_values_after,
    save_effect_output,
    effect_path_list,
    standardized_output,
    path,
    header,
    standardized,
):
    """
    Simulate phenotypes by loading all GRGs into RAM simultaneously.

    :param grg_files: List of paths to GRG files to be processed.
    :type grg_files: List[str]
    :param model: The distribution model from which effect sizes are drawn. Depends on the user's discretion.
    :param num_causal_per_file: Number of causal sites simulated for each file (same for each GRG).
    :param random_seed: The random seed used for causal mutation simulation.
    :param normalize_phenotype: Checks whether to normalize the phenotypes. Default: False.
    :param normalize_genetic_values_before_noise: Checks whether to normalize the genetic values
        prior to simulating environmental noise (True if yes). Depends on the user's discretion.
        Default: False.
    :param heritability: Takes in the h2 features to simulate environmental noise (set to None
        if the user prefers user-defined noise) and 1 is the user wants zero noise.
    :param user_defined_noise_parameters: Parameters used for simulating environmental noise
        taken in from the user.
    :param normalize_genetic_values_after: In the case where the h2 feature is not used, this
        checks whether the user wants genetic values normalized at the end (True if yes).
        Default: False.
    :param save_effect_output: This boolean parameter decides whether the effect sizes
        will be saved to a .par file using the standard output format. Default: False.
    :param effect_path: This parameter contains the path at which the .par output file will be saved.
        Default: None.
    :param standardized_output: This boolean parameter decides whether the phenotypes
        will be saved to a .phen file using the standard output format. Default: False.
    :param path: This parameter contains the path at which the .phen output file will be saved.
        Default: None.
    :param header: This boolean parameter decides whether the .phen output file contains column
        headers or not. Default value is False.
    :param standardized: This boolean parameters decides whether the simulation uses standardized
        genotypes.
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
            random_seed,
            normalize_genetic_values_before_noise,
            grg_name,
            effect_path=path,
            standardized=standardized,
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

    if heritability is not None:
        phenotypes = sim_env_noise(combined_genetic_df, h2=heritability)
        if normalize_phenotype:
            final_phenotypes = normalize(phenotypes)
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

    else:
        if check:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, mean=user_mean, std=user_cov
            )
        else:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, means=user_mean, cov=user_cov
            )

        if normalize_phenotype:
            final_phenotypes = normalize(
                phenotypes, normalize_genetic_values=normalize_genetic_values_after
            )
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

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
    normalize_phenotype,
    normalize_genetic_values_before_noise,
    population_prev,
    heritability,
    user_mean,
    user_cov,
    normalize_genetic_values_after,
    save_effect_output,
    effect_path_list,
    standardized_output,
    path,
    header,
    standardized,
):
    """
    Simulate phenotypes by processing GRGs sequentially to reduce memory usage.

    :param grg_files: List of paths to GRG files to be processed
    :type grg_files: List[str]
    :param model: The distribution model from which effect sizes are drawn. Depends on the user's discretion.
    :param num_causal_per_file: Number of causal sites simulated for each file (same for each GRG).
    :param random_seed: The random seed used for causal mutation simulation.
    :param normalize_phenotype: Checks whether to normalize the phenotypes. Default: False.
    :param normalize_genetic_values_before_noise: Checks whether to normalize the genetic values prior
        to simulating environmental noise (True if yes). Depends on the user's discretion.
        Default: False.
    :param heritability: Takes in the h2 features to simulate environmental noise (set to None if
        the user prefers user-defined noise) and 1 is the user wants zero noise.
    :param user_defined_noise_parameters: Parameters used for simulating environmental noise taken
        in from the user.
    :param normalize_genetic_values_after: In the case where the h2 feature is not used, this
        checks whether the user wants genetic values normalized at the end (True if yes).
        Default: False.
    :param save_effect_output: This boolean parameter decides whether the effect sizes
        will be saved to a .par file using the standard output format. Default value is False.
    :param effect_path: This parameter contains the path at which the .par output file will be
        saved. Default: None.
    :param standardized_output: This boolean parameter decides whether the phenotypes
        will be saved to a .phen file using the standard output format. Default: False.
    :param path: This parameter contains the path at which the .phen output file will be saved.
        Default: None.
    :param header: This boolean parameter decides whether the .phen output file contains column
        headers or not. Default: False.
    :param standardized: This boolean parameters decides whether the simulation uses standardized
        genotypes.
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
            random_seed,
            normalize_genetic_values_before_noise,
            grg_file,
            effect_path=path,
            standardized=standardized,
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

    if heritability is not None:
        phenotypes = sim_env_noise(combined_genetic_df, h2=heritability)
        if normalize_phenotype:
            final_phenotypes = normalize(phenotypes)
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

    else:
        if check:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, mean=user_mean, std=user_cov
            )
        else:
            phenotypes = sim_env_noise(
                combined_genetic_df, user_defined=True, means=user_mean, cov=user_cov
            )

        if normalize_phenotype:
            final_phenotypes = normalize(
                phenotypes, normalize_genetic_values=normalize_genetic_values_after
            )
        else:
            final_phenotypes = phenotype_class_to_df(phenotypes)

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
    model=grg_causal_mutation_model("normal", mean=0, var=1),
    num_causal_per_file=1000,
    random_seed=42,
    normalize_phenotype=False,
    load_all_ram=False,
    normalize_genetic_values_before_noise=False,
    heritability=None,
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
    standardized=False,
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
    normalize_phenotype: Checks whether to normalize the phenotypes. The default value is False.
    normalize_genetic_values_before_noise: Checks whether to normalize the genetic
    values prior to simulating environmental noise (True if yes). Depends on the
    user's discretion. Set to False by default.
    heritability: Takes in the h2 features to simulate environmental noise
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
    standardized: This boolean parameters decides whether the simulation uses standardized
    genotypes.

    Returns
    --------------------
    Pandas dataframe with resultant phenotypes. The dataframe contains the following:
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
            normalize_phenotype,
            normalize_genetic_values_before_noise,
            population_prev,
            heritability,
            user_mean,
            user_cov,
            normalize_genetic_values_after,
            save_effect_output,
            effect_path_list,
            standardized_output,
            path,
            header,
            standardized,
        )
    else:
        # Strategy 2: Process GRGs sequentially
        return sim_phenotypes_multi_grg_sequential(
            grg_files,
            model,
            num_causal_per_file,
            random_seed,
            normalize_phenotype,
            normalize_genetic_values_before_noise,
            population_prev,
            heritability,
            user_mean,
            user_cov,
            normalize_genetic_values_after,
            save_effect_output,
            effect_path_list,
            standardized_output,
            path,
            header,
            standardized,
        )
