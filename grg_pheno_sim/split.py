"""
This file contains functions to split the phenotypic dataframes for multivariate simulation.
=======
"""


def split_effect_sizes(effect_sizes_df, return_print=False, return_list=False):
    """
    Function to split effect sizes dataframe in the multivariate case.

    Parameters
    -----------------------
    effect_sizes_df: The input effect sizes pandas dataframe to be split.
    return_print: This boolean parameter prints the split dataframes if set to True.
    Default value is False.
    return_list: This boolean parameter returns a list of the split dataframes if set
    to True. Default value is False.
    """

    df_dict = {
        k: v.sort_values("mutation_id")
        for k, v in effect_sizes_df.groupby("causal_mutation_id")
    }

    num = len(df_dict)
    df_list = []

    if return_print == True:
        for i in range(num):
            print(df_dict[i])
    elif return_list == True:
        for i in range(num):
            df_list.append(df_dict[i])
        return df_list


def split_genetic_values(genetic_values_df, return_print=False, return_list=False):
    """
    Function to split genetic values dataframe in the multivariate case.

    Parameters
    -----------------------
    genetic_values_df: The input genetic values pandas dataframe to be split.
    return_print: This boolean parameter prints the split dataframes if set to True.
    Default value is False.
    return_list: This boolean parameter returns a list of the split dataframes if set
    to True. Default value is False.
    """

    df_dict = {
        k: v.sort_values("sample_node_id")
        for k, v in genetic_values_df.groupby("causal_mutation_id")
    }

    num = len(df_dict)
    df_list = []

    if return_print == True:
        for i in range(num):
            print(df_dict[i])
    elif return_list == True:
        for i in range(num):
            df_list.append(df_dict[i])
        return df_list


def split_phenotypes(phenotype_df, return_print=False, return_list=False):
    """
    Function to split phenotype dataframe in the multivariate case.

    Parameters
    -----------------------
    phenotype_df: The input phenotype pandas dataframe to be split.
    return_print: This boolean parameter prints the split dataframes if set to True.
    Default value is False.
    return_list: This boolean parameter returns a list of the split dataframes if set
    to True. Default value is False.
    """

    df_dict = {
        k: v.sort_values("individual_id")
        for k, v in phenotype_df.groupby("causal_mutation_id")
    }

    num = len(df_dict)
    df_list = []

    if return_print == True:
        for i in range(num):
            print(df_dict[i])
    elif return_list == True:
        for i in range(num):
            df_list.append(df_dict[i])
        return df_list
