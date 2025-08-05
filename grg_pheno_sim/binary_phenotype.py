"""
This file simulates binary phenotypes on GRGs by using the usual simulation methods
and then converting continuous phenotypes to binary phenotypes.
=======
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from grg_pheno_sim.phenotype import sim_phenotypes_StdOp
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
from grg_pheno_sim.phenotype import convert_to_phen
from grg_pheno_sim.ops_scipy import SciPyStdXOperator as _SciPyStdXOperator
from grg_pheno_sim.phenotype import allele_frequencies_new

def sim_binary_phenotypes(
    grg,
    population_prevalence,
    model=grg_causal_mutation_model("normal", mean=0, var=1),
    num_causal=1000,
    random_seed=42,
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
    standardized = False
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
    standardized: This boolean parameters decides whether the simulation uses standardized genotypes.


    Returns
    --------------------
    Pandas dataframe with resultant binary phenotypes. The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    """
    # if standardized:               

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

    if heritability is not None:
        phenotypes = sim_env_noise(individual_genetic_values, h2=heritability)
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
    heritability: Takes in the h2 features to simulate environmental noise
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

    if heritability is not None:
        phenotypes = sim_env_noise(individual_genetic_values, h2=heritability)
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

def sim_binary_phenotypes_standOp(
    grg,
    population_prevalence: float,
    heritability: float,
    num_causal: int = 1000,
    random_seed: int = 42,
    standardized_output: bool = False,
    path: str = None,
    header: bool = False,
) -> pd.DataFrame:
    """
    Simulate binary phenotypes using the standardized‐operator pipeline.

    Parameters
    ----------
    grg
        Your GRG object.
    population_prevalence
        Prevalence k (e.g. 0.1 for 10% case rate).
    heritability
        Narrow‐sense h² used in the standardized pipeline.
    num_causal
        Number of causal SNPs to draw.
    random_seed
        Seed for causal effects & noise.
    standardized_output
        If True, writes out a `.phen` file.
    path
        Path for the `.phen` output (tab‐delimited).
    header
        Whether to include a header line in the `.phen`.

    Returns
    -------
    A DataFrame with columns
      individual_id, genetic_value, causal_mutation_id,
      environmental_noise, phenotype
    where `phenotype` is 0/1.
    """

    # 1) Get continuous phenotypes (G + E)
    df_cont = sim_phenotypes_StdOp(
        grg,
        heritability=heritability,
        num_causal=num_causal,
        random_seed=random_seed
    )

    # 2) Threshold to binary
    k = population_prevalence
    T = stats.norm.ppf(1.0 - k)
    df_cont["phenotype"] = (df_cont["phenotype"] >= T).astype(int)
    
    if standardized_output == True:
        convert_to_phen(df_cont, path, include_header=header)

    return df_cont

def sim_binary_phenotypes_custom_stdOp(
    grg,
    input_effects,
    population_prevalence: float,
    heritability: float,
    random_seed: int = 42,
    save_effect_output: bool = False,
    effect_path: str = None,
    standardized_output: bool = False,
    path: str = None,
    header: bool = False,
) -> pd.DataFrame:
    """
    Simulate binary phenotypes with custom effect sizes using the standardized‐operator.

    Parameters
    ----------
    grg
        Your GRG object.
    input_effects
        Custom effects: dict{mut_id:beta}, or list, or DataFrame with
        ["mutation_id","effect_size"].
    population_prevalence
        Case‐rate k (e.g. 0.1 → 10% cases).
    heritability
        Narrow‐sense h².
    random_seed
        RNG seed for reproducibility.
    save_effect_output
        If True, writes out a `.par` to `effect_path`.
    effect_path
        Path for `.par` (if save_effect_output).
    standardized_output
        If True, writes out a `.phen` to `path`.
    path
        Path for `.phen` (if standardized_output).
    header
        Include header line in the `.phen`.

    Returns
    -------
    pd.DataFrame
        Columns: individual_id, genetic_value, causal_mutation_id,
                 environmental_noise, phenotype (0/1).
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
    
    M = grg.num_mutations
    beta = np.zeros(M, dtype=float)
    beta[causal_mutation_df["mutation_id"].astype(int).values] = causal_mutation_df["effect_size"].values


    freqs = allele_frequencies_new(grg)
    std_op = _SciPyStdXOperator(
        grg,
        direction=pygrgl.TraversalDirection.UP,
        freqs=freqs,
        haploid=False
    )
    G = std_op._matmat(beta.reshape(-1, 1)).squeeze()

    df = pd.DataFrame({
        "individual_id": np.arange(grg.num_individuals, dtype=int),
        "genetic_value": G,
        "causal_mutation_id": 0
    })

    gvar = df["genetic_value"].var(ddof=1)
    noise_var = gvar * (1.0 / heritability - 1.0)
    rng = np.random.default_rng(random_seed)
    
    df["environmental_noise"] = rng.normal(
        0.0, np.sqrt(noise_var), size=len(df)
    )

    T = stats.norm.ppf(1.0 - population_prevalence)
    df["phenotype"] = (df["genetic_value"] + df["environmental_noise"] >= T).astype(int)

    if standardized_output:
        convert_to_phen(df, path, include_header=header)

    return df
