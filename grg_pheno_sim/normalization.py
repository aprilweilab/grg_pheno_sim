"""
This file normalizes the phenotypic data for the individuals. It includes functions
for quantile normalization as well.
=======
"""

import pandas as pd
from scipy.stats import norm


def normalize_genetic_val(phenotype_df):
    """
    Function to normalize the genetic values post environmental noise simulation.
    """
    mean_df = phenotype_df["genetic_value"].mean()
    std_dev_df = phenotype_df["genetic_value"].std()
    phenotype_df["genetic_value"] = (
        phenotype_df["genetic_value"] - mean_df
    ) / std_dev_df

    return phenotype_df


def normalize_individual_phenotypes(
    phenotype_df, h2_curr, user_noise, normalize_genetic_values
):
    """
    Function to normalize individual phenotype dataframes based on simulation requirements.
    """
    old_mean = phenotype_df["phenotype"].mean()
    old_std = phenotype_df["phenotype"].std()

    phenotype_df["phenotype"] = (phenotype_df["phenotype"] - old_mean) / old_std

    old_genetic_mean = phenotype_df["genetic_value"].mean()
    old_noise_mean = phenotype_df["environmental_noise"].mean()

    if (h2_curr != 1 and user_noise == False) or user_noise == True:

        # scale both genetic and noise by subtracting E[g], E[noise] and divide by phenotype std in both cases
        phenotype_df["genetic_value"] = (
            phenotype_df["genetic_value"] - old_genetic_mean
        ) / old_std
        phenotype_df["environmental_noise"] = (
            phenotype_df["environmental_noise"] - old_noise_mean
        ) / old_std

    elif h2_curr == 1 and user_noise == False:
        if normalize_genetic_values == True:
            phenotype_df = normalize_genetic_val(phenotype_df)
        else:
            return phenotype_df

    return phenotype_df


def normalize(phenotypes, normalize_genetic_values=False):
    """
    Function to scale the genetic values such that the condition
    Var(genetic_value) + Var(environmental_noise) = Var(phenotype)
    is fulfilled. Calls the function `normalize_individual_phenotypes` as needed.

    Parameters
    -------------
    phenotypes: The input pandas dataframe containing the computed phenotypes.
    normalize_genetic_values: The boolean parameter that decides whether the
    genetic values will be normalized as well in the resultant dataframe. Default
    value is False.

    Returns
    --------------------
    Pandas dataframe with resultant normalized phenotypes. The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    """

    phenotype_df = phenotypes.phenotype_df
    h2 = phenotypes.h2
    user_noise = phenotypes.user_noise

    causal_mutation_id_check = phenotype_df["causal_mutation_id"].unique()

    if len(causal_mutation_id_check) == 1:
        normalized_phenotype_df = normalize_individual_phenotypes(
            phenotype_df, h2[0], user_noise, normalize_genetic_values
        )

    else:
        df_dict = {
            k: v.sort_values("individual_id")
            for k, v in phenotype_df.groupby("causal_mutation_id")
        }

        num = len(df_dict)

        normalized_dict = {i: None for i in range(num)}

        for i in range(num):
            pheno_df = df_dict[i]

            normalized_df = normalize_individual_phenotypes(
                pheno_df, h2[i], user_noise, normalize_genetic_values
            )
            normalized_dict[i] = normalized_df

        interleaved_rows = []
        length = len(normalized_dict[0])

        for i in range(length):
            for key in sorted(normalized_dict.keys()):
                interleaved_rows.append(normalized_dict[key].iloc[i])

        normalized_phenotype_df = pd.concat(interleaved_rows, axis=1).transpose()

        normalized_phenotype_df.reset_index(drop=True, inplace=True)

    normalized_phenotype_df["individual_id"] = normalized_phenotype_df[
        "individual_id"
    ].astype(int)
    normalized_phenotype_df["causal_mutation_id"] = normalized_phenotype_df[
        "causal_mutation_id"
    ].astype(int)

    return normalized_phenotype_df


def quantile_normalize_to_normal(column, epsilon=1e-10):
    """
    Function to quantile normalize a dataframe column to the normal distribution.
    Adds epsilon to avoid values exactly 0 or 1 in ranks.
    """
    ranks = column.rank(method="average", pct=True)
    ranks = ranks.clip(
        lower=epsilon, upper=1 - epsilon
    )  # adjust ranks to avoid 0 and 1
    normal_values = norm.ppf(ranks)

    return pd.Series(normal_values, index=column.index)


def quantile_normalize(phenotype_df, phenotype_normalize=True, normalize_both=False):
    """
    Function to quantile normalize phenotypic dataframes. Normalizes either the phenotypes or genetic values.
    Calls on function `quantile_normalize_to_normal` as needed.

    Parameters
    -------------
    phenotypes: The input pandas dataframe containing the computed phenotypes.
    phenotype_normalize: The boolean parameter that decides whether the
    phenotypes will be normalized in the resultant dataframe. Default
    value is True.
    normalize_both: The boolean parameter that decides whether the
    genetic values will be normalized as well in the resultant dataframe. Default
    value is False.

    Returns
    --------------------
    Pandas dataframe with quantile normalized phenotypes added as a new column.
    The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    `normalized_phenotype`
    """

    causal_mutation_id_check = phenotype_df["causal_mutation_id"].unique()

    if len(causal_mutation_id_check) == 1:
        if normalize_both == True:
            phenotype_df["normalized_genetic_value"] = quantile_normalize_to_normal(
                phenotype_df["genetic_value"]
            )
            phenotype_df["normalized_phenotype"] = quantile_normalize_to_normal(
                phenotype_df["phenotype"]
            )
        elif phenotype_normalize == True:
            phenotype_df["normalized_phenotype"] = quantile_normalize_to_normal(
                phenotype_df["phenotype"]
            )
        else:
            phenotype_df["normalized_genetic_value"] = quantile_normalize_to_normal(
                phenotype_df["genetic_value"]
            )

        return phenotype_df

    else:
        df_dict = {
            k: v.sort_values("individual_id")
            for k, v in phenotype_df.groupby("causal_mutation_id")
        }

        num = len(df_dict)

        quantile_normalized_dict = {i: None for i in range(num)}

        for i in range(num):
            phenotype_df = df_dict[i]
            if normalize_both == True:
                phenotype_df["normalized_genetic_value"] = quantile_normalize_to_normal(
                    phenotype_df["genetic_value"]
                )
                phenotype_df["normalized_phenotype"] = quantile_normalize_to_normal(
                    phenotype_df["phenotype"]
                )
            elif phenotype_normalize == True:
                phenotype_df["normalized_phenotype"] = quantile_normalize_to_normal(
                    phenotype_df["phenotype"]
                )
            else:
                phenotype_df["normalized_genetic_value"] = quantile_normalize_to_normal(
                    phenotype_df["genetic_value"]
                )

            quantile_normalized_dict[i] = phenotype_df

        interleaved_rows = []
        length = len(quantile_normalized_dict[0])

        for i in range(length):
            for key in sorted(quantile_normalized_dict.keys()):
                interleaved_rows.append(quantile_normalized_dict[key].iloc[i])

        quantile_normalized_phenotype_df = pd.concat(
            interleaved_rows, axis=1
        ).transpose()

        quantile_normalized_phenotype_df.reset_index(drop=True, inplace=True)

        quantile_normalized_phenotype_df["individual_id"] = (
            quantile_normalized_phenotype_df["individual_id"].astype(int)
        )
        quantile_normalized_phenotype_df["causal_mutation_id"] = (
            quantile_normalized_phenotype_df["causal_mutation_id"].astype(int)
        )

        return quantile_normalized_phenotype_df
