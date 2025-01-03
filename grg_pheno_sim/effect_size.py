"""
This file simulates effect sizes on GRGs by using the models defined in `model.py`
=======
"""

import numpy as np
import pandas as pd
import pygrgl


class GRGCausalMutationSimulator:
    """
    Simulator class to simulate effect sizes of causal mutations.
    """

    def __init__(self, grg, causal_sites, model, rng):
        """
        The initializer takes in a GRG, the causal sites, simulation model and rng model.
        """
        self.grg = grg
        self.causal_sites = causal_sites
        self.model = model
        self.rng = rng

    def sim_effect_sizes(self):
        """
        Simulate effect sizes based on the model and the number of causal sites.
        """
        num_causal = len(self.causal_sites)
        return self.model.sim_effect_size(num_causal, self.rng)

    def run(self):
        """
        Run the effect size simulation and return results in a DataFrame.
        """
        effect_sizes = self.sim_effect_sizes()
        if effect_sizes.ndim == 1:
            effect_sizes = effect_sizes[:, np.newaxis]

        num_traits = self.model.num_trait
        causal_mutation_data = {
            "mutation_id": np.repeat(self.causal_sites, num_traits),
            "effect_size": effect_sizes.ravel(),
            "causal_mutation_id": np.tile(
                np.arange(num_traits), len(self.causal_sites)
            ),
        }
        df = pd.DataFrame(causal_mutation_data)

        # sort by mutation_id first and then by causal_mutation_id to ensure the order
        df = df.sort_values(by=["mutation_id", "causal_mutation_id"]).reset_index(
            drop=True
        )

        return df


def sim_grg_causal_mutation(
    grg, model, *, num_causal=None, causal_sites=None, random_seed=None
):
    """
    Function to initialize the simulator and run the simulation.
    """
    rng = np.random.default_rng(random_seed)
    if num_causal is not None and causal_sites is not None:
        raise ValueError("Specify either num_causal or causal_sites, not both.")
    if causal_sites is None:
        if num_causal is None:
            num_causal = 1
        num_sites = grg.num_mutations
        causal_sites = rng.choice(num_sites, size=num_causal, replace=False)

    simulator = GRGCausalMutationSimulator(grg, causal_sites, model, rng)
    return simulator.run()


def allele_frequency_dot(grg, mutations):
    """
    Function to calculate allele frequencies for a single causal mutation using pygrgl.dot_product.
    """
    num_samples = grg.num_samples

    input_vector = np.full(num_samples, 1 / num_samples, dtype=np.float64)

    output_vector = pygrgl.dot_product(grg, input_vector, pygrgl.TraversalDirection.UP)

    return output_vector


def allele_frequencies(grg, effects):
    """
    Function to handle computation of allele frequencies.
    """

    if (
        effects["causal_mutation_id"].sum() == 0
    ):  # base case for univariate causal mutation simulation
        mutations = effects["mutation_id"].tolist()
        frequencies = allele_frequency_dot(grg, mutations)

        final_frequencies = []

        for mutation in mutations:
            final_frequencies.append(frequencies[mutation])

        return final_frequencies

    else:  # for multivariate causal mutation simulation
        df_dict = {
            k: v.sort_values("mutation_id")
            for k, v in effects.groupby("causal_mutation_id")
        }

        num = len(df_dict)

        frequencies = []

        sample_muts = []

        for i in range(num):
            temp_effects = df_dict[i]
            mutations = temp_effects["mutation_id"].tolist()
            sample_muts = mutations
            frequencies.append(allele_frequency_dot(grg, mutations))

        interleaved_rows = []

        for id in sample_muts:
            for i in range(num):
                temp_frequency = frequencies[i]
                interleaved_rows.append(temp_frequency[id])

        return interleaved_rows


def convert_to_effect_output(effects_df, grg, path):
    """
    Function to output and save standardized effect sizes to a .par file.
    """
    if path is None:
        raise ValueError("Output path must be defined")

    mutations = effects_df["mutation_id"].tolist()

    ref_alleles, alleles, positions = [], [], []

    frequencies_allele = allele_frequencies(grg, effects_df)

    for mutation in mutations:
        mut = grg.get_mutation_by_id(mutation)
        ref_alleles.append(mut.ref_allele)
        alleles.append(mut.allele)
        positions.append(mut.position)

    effects_data = {
        "mutation_id": mutations,
        "AlternateAllele": alleles,
        "Position": positions,
        "RefAllele": ref_alleles,
        "Frequency": frequencies_allele,
        "Effect": effects_df["effect_size"],
    }

    df_effects = pd.DataFrame(effects_data)

    df_effects.to_csv(path, sep="\t", index=False, header=True)


def additive_sim_grg_causal_mutation(grg, causal_mutation_df, causal_id):
    """
    Function to compute effect sizes using matrix operations via dot product.
    Takes advantage of the GRG structure to efficiently calculate cumulative effects.

    Args:
        grg: Genotype relationship graph object
        causal_mutation_df: DataFrame containing mutation IDs and their effect sizes
        causal_id: ID of the causal mutation being analyzed

    Returns:
        DataFrame containing sample nodes and their cumulative genetic values
    """
    # Create mutation effect size vector using GRG's actual mutation count
    num_mutations = grg.num_mutations  # Get actual number of mutations from GRG
    mutation_vector = np.zeros(num_mutations, dtype=np.float64)

    # Fill the mutation vector with effect sizes
    for _, row in causal_mutation_df.iterrows():
        mutation_idx = int(row["mutation_id"])
        if mutation_idx < num_mutations:  # Add safety check
            mutation_vector[mutation_idx] = float(row["effect_size"])
        else:
            raise ValueError(
                f"Mutation ID {mutation_idx} exceeds the number of mutations in GRG ({num_mutations})"
            )

    # Use dot_product with DOWN direction to get effect sizes for all nodes
    sample_effects = pygrgl.dot_product(
        grg=grg, input=mutation_vector, direction=pygrgl.TraversalDirection.DOWN
    )

    # Create output DataFrame with only sample nodes
    samples_list = grg.get_sample_nodes()
    sample_effects_df = pd.DataFrame(
        {
            "sample_node_id": samples_list,
            "genetic_value": [sample_effects[node] for node in samples_list],
            "causal_mutation_id": causal_id,
        }
    )

    return sample_effects_df


def additive_effect_sizes(grg, causal_mutation_df):
    """
    Function to handle pass down of values. Calls helper function `additive_sim_grg_causal_mutation` to handle
    passing effect sizes down the GRG for each individual causal mutation.

    Parameters
    --------------------
    grg: The input GRG using which the genetic values must be computed.
    causal_mutation_df: The input pandas dataframe containing the effect sizes
    attached to their corresponding mutation IDs.

    Returns
    --------------------
    Pandas dataframe with genetic values for sample nodes.
    """
    if (
        causal_mutation_df["causal_mutation_id"].sum() == 0
    ):  # base case for univariate causal mutation simulation
        samples_effect_sizes = additive_sim_grg_causal_mutation(
            grg, causal_mutation_df, 0
        )

        return samples_effect_sizes

    else:  # for multivariate causal mutation simulation
        df_dict = {
            k: v.sort_values("mutation_id")
            for k, v in causal_mutation_df.groupby("causal_mutation_id")
        }

        num = len(df_dict)

        sample_nodes_dict = {i: None for i in range(num)}

        for i in range(num):
            sample_nodes_df = additive_sim_grg_causal_mutation(grg, df_dict[i], i)
            sample_nodes_dict[i] = sample_nodes_df

        interleaved_rows = []
        length = len(sample_nodes_dict[0])

        for i in range(length):
            for key in sorted(sample_nodes_dict.keys()):
                interleaved_rows.append(sample_nodes_dict[key].iloc[i])

        samples_effect_sizes = pd.concat(interleaved_rows, axis=1).transpose()

        samples_effect_sizes.reset_index(drop=True, inplace=True)

        samples_effect_sizes["sample_node_id"] = samples_effect_sizes[
            "sample_node_id"
        ].astype(int)
        samples_effect_sizes["causal_mutation_id"] = samples_effect_sizes[
            "causal_mutation_id"
        ].astype(int)

        return samples_effect_sizes


def singular_samples_to_individuals(additive_causal_mutation_df):
    """
    Function used to convert genetic values for sample nodes to individuals for a single
    causal mutation.
    """

    if len(additive_causal_mutation_df) % 2 != 0:
        additive_causal_mutation_df = additive_causal_mutation_df.iloc[:-1]

    individual_df = {
        "individual_id": range(len(additive_causal_mutation_df) // 2),
        "genetic_value": additive_causal_mutation_df["genetic_value"]
        .groupby(additive_causal_mutation_df.index // 2)
        .sum(),
        "causal_mutation_id": additive_causal_mutation_df["causal_mutation_id"][
            ::2
        ].values,
    }

    individual_df_effect_sizes = pd.DataFrame(individual_df)

    return individual_df_effect_sizes


def multiple_samples_to_individuals(additive_causal_mutation_df):
    """
    Function used to convert genetic values for sample nodes to individuals for a single
    causal mutation, but in the multivariate case.
    """
    additive_causal_mutation_df["pair_id"] = (
        additive_causal_mutation_df["sample_node_id"] // 2
    )

    individual_df = (
        additive_causal_mutation_df.groupby("pair_id")
        .agg({"genetic_value": "sum", "causal_mutation_id": "first"})
        .reset_index()
    )

    individual_df.rename(columns={"pair_id": "individual_id"}, inplace=True)

    return individual_df


def samples_to_individuals(additive_causal_mutation_df):
    """
    Function to combine effect sizes for pairs of samples nodes to effect sizes for individuals. Calls on
    `singular_samples_to_individuals` and `multiple_samples_to_individuals` as needed.

    Parameters
    ------------------
    additive_causal_mutation_df: The input pandas dataframe containing genetic values
    for each sample node.

    Returns
    --------------------
    Pandas dataframe with genetic values for individuals.
    """
    if (
        additive_causal_mutation_df["causal_mutation_id"].sum() == 0
    ):  # base case for univariate causal mutation

        individuals = singular_samples_to_individuals(additive_causal_mutation_df)

        return individuals

    else:
        df_dict = {
            k: v.sort_values("sample_node_id")
            for k, v in additive_causal_mutation_df.groupby("causal_mutation_id")
        }

        num = len(df_dict)

        individual_nodes_dict = {i: None for i in range(num)}

        for i in range(num):
            individual_nodes_df = multiple_samples_to_individuals(df_dict[i])
            individual_nodes_dict[i] = individual_nodes_df

        interleaved_rows = []
        length = len(individual_nodes_dict[0])

        for i in range(length):
            for key in sorted(individual_nodes_dict.keys()):
                interleaved_rows.append(individual_nodes_dict[key].iloc[i])

        individuals = pd.concat(interleaved_rows, axis=1).transpose()

        individuals.reset_index(drop=True, inplace=True)

        individuals["individual_id"] = individuals["individual_id"].astype(int)
        individuals["causal_mutation_id"] = individuals["causal_mutation_id"].astype(
            int
        )

        return individuals


def normalize_individual_genetic_values(individual_df, mean=0, var=1):
    """
    Function to normalize genetic values for individual causal mutations.
    """
    if var <= 0:
        raise ValueError("Variance must be greater than 0.")

    mean_df = individual_df["genetic_value"].mean()
    std_dev_df = individual_df["genetic_value"].std()
    individual_df["genetic_value"] = (
        individual_df["genetic_value"] - mean_df
    ) / std_dev_df

    return individual_df


def normalize_genetic_values(individuals_df, mean=0, var=1):
    """
    Function to normalize genetic values for any number of causal mutations. Calls on
    `normalize_individual_genetic_values` for computation.

    Parameters
    ------------------
    individuals_df: The pandas dataframe containing genetic values combined for each
    individual.
    mean: The resultant mean required post normalization. Defaults to standard Gaussian.
    var: The resultant variance required post normalization. Defaults to standard Gaussian.

    Returns
    --------------------
    Pandas dataframe with normalized genetic values for individuals.
    """

    if (
        individuals_df["causal_mutation_id"].sum() == 0
    ):  # base case for univariate causal mutation
        individuals_df = normalize_individual_genetic_values(individuals_df, mean, var)

        return individuals_df

    else:
        df_dict = {
            k: v.sort_values("individual_id")
            for k, v in individuals_df.groupby("causal_mutation_id")
        }

        num = len(df_dict)

        normalized_nodes_dict = {i: None for i in range(num)}

        for i in range(num):
            individual_nodes_df = normalize_individual_genetic_values(df_dict[i])
            normalized_nodes_dict[i] = individual_nodes_df

        interleaved_rows = []

        length = len(normalized_nodes_dict[0])

        for i in range(length):
            for key in sorted(normalized_nodes_dict.keys()):
                interleaved_rows.append(normalized_nodes_dict[key].iloc[i])

        individuals_df = pd.concat(interleaved_rows, axis=1).transpose()

        individuals_df.reset_index(drop=True, inplace=True)

        individuals_df["individual_id"] = individuals_df["individual_id"].astype(int)
        individuals_df["causal_mutation_id"] = individuals_df[
            "causal_mutation_id"
        ].astype(int)

        return individuals_df
