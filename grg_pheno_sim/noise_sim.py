"""
This file simulates the environmental noise for the individual genetic values.
=======
"""

import numpy as np
from grg_pheno_sim.split import split_phenotypes


class GrgPhenotype:
    """
    Class to store the final phenotypes dataframe along with the h2 information encoded, used to simulate noise.

    Parameters
    ----------
    phenotype_df : pandas.DataFrame
         DataFrame containing genetic values of individuals, noise, and phenotype. Must include individual_id, genetic_value, environmental_noise, phenotype, and causal_mutation_id.
     h2 : float or array-like
         Narrow-sense heritability.
     user_noise : bool
         Encodes whether the environmental noise is customized to genetic values (False) or individually defined by the user (True).
    """

    def __init__(self, phenotype_df, h2, user_noise):
        self.phenotype_df = phenotype_df
        self.h2 = h2
        self.user_noise = user_noise

    def get_h2(self):
        if self.user_noise == False:
            return self.h2

        else:
            phen = self.phenotype_df
            causal_mutation_id = phen["causal_mutation_id"].unique()

            if len(causal_mutation_id) == 1:

                genetic_var = phen["genetic_value"].var()
                noise_var = phen["environmental_noise"].var()
                phenotype_var = phen["phenotype"].var()

                new_h2 = (genetic_var + noise_var) / phenotype_var

            else:
                phen_list = split_phenotypes(self.phenotype_df, return_list=True)

                genetic_vars, noise_vars, phenotype_vars = [], [], []

                for i in phen_list:
                    genetic_vars.append(i["genetic_value"].var())
                    noise_vars.append(i["environmental_noise"].var())
                    phenotype_vars.append(i["phenotype"].var())

                genetic_vars = np.array(genetic_vars)
                noise_vars = np.array(noise_vars)
                phenotype_vars = np.array(phenotype_vars)

                new_h2 = (genetic_vars + noise_vars) / phenotype_vars

            return new_h2


class GrgEnvSimulator:
    """Simulator class to simulate environmental noise of individuals in the GRG based on genetic values and causal mutations.

    Parameters
    ----------
    genetic_value_df : pandas.DataFrame
        DataFrame containing genetic values of individuals. Must include individual_id, genetic_value, and causal_mutation_id.
    h2 : float or array-like
        Narrow-sense heritability.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(self, genetic_value_df, h2, random_seed):
        self.genetic_value_df = genetic_value_df[
            ["causal_mutation_id", "individual_id", "genetic_value"]
        ]
        self.h2 = h2
        self.rng = np.random.default_rng(random_seed)

    def sim_env_univariate(self, var, h2):
        """
        Simulate environmental noise based on variance and heritability.
        """
        env_std = np.sqrt((1 - h2) / h2 * var)

        return self.rng.normal(0.0, env_std)

    def sim_env_multivariate(self, var_array, h2_array):
        """
        Simulate environmental noise for multiple mutations based on their variances and heritabilities.
        """
        env_std = np.sqrt((1 - h2_array) / h2_array * var_array)

        return self.rng.normal(0.0, env_std)

    def sim_environment_univariate(self, tailored=False, user_mean=None, user_std=None):
        """
        Simulate environmental values based on genetic values and heritability.
        """
        df = self.genetic_value_df.copy()

        if tailored == False:
            h2_array = np.take(self.h2, self.genetic_value_df.causal_mutation_id)

            grouped = df.groupby("causal_mutation_id")["genetic_value"]
            var_array = grouped.transform("var")

            df["environmental_noise"] = self.sim_env_univariate(var_array, h2_array)

        else:
            env_noise = self.rng.normal(loc=user_mean, scale=user_std, size=len(df))
            df["environmental_noise"] = env_noise

        df["phenotype"] = df["genetic_value"] + df["environmental_noise"]

        return df

    def sim_environment_multivariate(
        self, num_causal, means=None, cov=None, tailored=False
    ):
        """
        Simulate environmental values based on genetic values and heritability, for the multivariate case.
        """

        df = self.genetic_value_df.copy()

        if tailored == False:
            h2_array = np.take(self.h2, self.genetic_value_df.causal_mutation_id)

            grouped = df.groupby("causal_mutation_id")["genetic_value"]
            var_array = grouped.transform("var")

            env_noise = self.sim_env_multivariate(var_array, h2_array)

            df["environmental_noise"] = env_noise

        else:
            env_noise = self.rng.multivariate_normal(means, cov, size=int(num_causal))
            df["environmental_noise"] = env_noise.ravel()

        df["phenotype"] = df["genetic_value"] + df["environmental_noise"]

        return df


def sim_env_noise(
    genetic_value_df,
    *,
    h2=None,
    random_seed=None,
    user_defined=False,
    mean=None,
    std=None,
    means=None,
    cov=None
):
    """
    Overarching function to simulate environmental noise. Call the univariate/multivariate
    noise simulation methods from the GrgEnvSimulator class as required.

    Parameters
    ------------------
    genetic_value_df: This is the pandas dataframe containing the genetic values.
    h2: This is the narrow sense h2 heritability factor. Default value is None.
    random_seed: This is the random seed used for noise simulation.
    user_defined: This boolean parameter evaluates whether the noise will be simulated
    from distributions defined by the user. Default value is False.

    The following two parameters are only relevant for user-defined univariate noise simulation:
    mean: This is the user defined mean for noise simulation. Default value is None.
    std: This is the user defined standard deviation for noise simulation. Default value is None.

    The following two parameters are only relevant for user-defined multivariate noise simulation:
    means: This is the user defined means array for noise simulation. Default value is None.
    cov: This is the user defined covariance matrix for noise simulation. Default value is None.

    Returns
    --------------------
    Pandas dataframe with environmental noise and resultant phenotypes. The dataframe contains the following:
    `causal_mutation_id`
    `individual_id`
    `genetic_value`
    `environmental_noise`
    `phenotype`
    """

    causal_mutation_id = genetic_value_df["causal_mutation_id"].unique()

    if (
        np.min(causal_mutation_id) != 0
        or np.max(causal_mutation_id) != len(causal_mutation_id) - 1
    ):
        raise ValueError("causal_mutation_id must be consecutive and start from 0")

    h2 = [1] if h2 is None else h2

    if h2 is not None and len(causal_mutation_id) == 1:
        h2 = np.ones(len(causal_mutation_id)) * h2

    if h2 == [1] and len(causal_mutation_id) > 1:
        h2 = np.ones(len(causal_mutation_id)) * h2

    if len(h2) != len(causal_mutation_id):
        raise ValueError("Length of h2 must match the number of causal mutations")

    if np.min(h2) <= 0 or np.max(h2) > 1:
        raise ValueError("Narrow-sense heritability must be 0 < h2 <= 1")

    simulator = GrgEnvSimulator(
        genetic_value_df=genetic_value_df,
        h2=h2,
        random_seed=random_seed,
    )

    if len(causal_mutation_id) == 1:

        if user_defined == True:
            phenotype_df = simulator.sim_environment_univariate(
                tailored=user_defined, user_mean=mean, user_std=std
            )
        else:
            phenotype_df = simulator.sim_environment_univariate()

    else:
        num_causal = len(genetic_value_df) / len(causal_mutation_id)
        if user_defined == True:
            phenotype_df = simulator.sim_environment_multivariate(
                tailored=user_defined, means=means, cov=cov, num_causal=num_causal
            )
        else:
            phenotype_df = simulator.sim_environment_multivariate(num_causal)

    phenotypes = GrgPhenotype(
        phenotype_df=phenotype_df,
        h2=h2,
        user_noise=user_defined,
    )

    return phenotypes
