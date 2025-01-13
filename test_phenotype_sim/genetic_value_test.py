def split_sample_df(sample_nodes_df):
    df_dict = {
        k: v.sort_values("sample_node_id")
        for k, v in sample_nodes_df.groupby("causal_mutation_id")
    }

    num = len(df_dict)

    return df_dict, num


def split_causal_mutation_df(trait_df):
    df_dict = {
        k: v.sort_values("mutation_id")
        for k, v in trait_df.groupby("causal_mutation_id")
    }

    return df_dict


def split_normalized_genetic_values(normalized_df):
    df_dict = {
        k: v.sort_values("individual_id")
        for k, v in normalized_df.groupby("causal_mutation_id")
    }

    return df_dict, len(df_dict)
