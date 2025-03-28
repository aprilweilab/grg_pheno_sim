"""
This file defines the distribution models used for causal mutation simulation.
=======
"""

from abc import ABCMeta, abstractmethod
import numpy as np

from grg_pheno_sim.validation import check_type, check_int


class GRGCausalMutationModel(metaclass=ABCMeta):
    """
    Superclass of the causal mutation model for Genotype Representation Graphs (GRGs).

    Attributes
    ----------
    name : str
        Name of the causal mutation model.
    num_trait : int
        Number of causal mutations to be simulated.

    Methods
    -------
    sim_effect_size(self, num_causal, rng):
        Simulates effect sizes based on the model. Must be implemented by all subclasses.
    """

    def __init__(self, name, num_trait=1):
        """
        Initializes the GRGCausalMutationModel with a name and number of Causal Mutations.

        Parameters
        ----------
        name : str
            The name of the Causal Mutation model.
        num_trait : int
            Number of causal mutations this model will simulate.
        """
        self.name = name
        self.num_trait = num_trait

    def check_parameter(self, num_causal, rng):
        """
        Validates the parameters for effect size simulation.

        Parameters
        ----------
        num_causal : int
            The number of causal sites to consider for Causal Mutation simulation.
        rng : np.random.Generator
            A random number generator instance.

        Returns
        -------
        num_causal : int
            The validated number of causal sites.
        """
        num_causal = check_int(num_causal, "num_causal", minimum=1)
        check_type(rng, "rng", np.random.Generator)
        return num_causal

    @abstractmethod
    def sim_effect_size(self, num_causal, rng):
        """
        Abstract method to simulate effect sizes. This method must be implemented
        by subclasses to define how effect sizes are generated based on the model.

        Parameters
        ----------
        num_causal : int
            The number of causal mutations for which to simulate effects.
        rng : np.random.Generator
            A random number generator for stochastic processes.

        Returns
        -------
        np.array
            An array of simulated effect sizes.
        """
        pass


class GRGCausalMutationModelNormal(GRGCausalMutationModel):
    """
    Normal distribution Causal Mutation model.

    Parameters
    ----------
    mean : float
        Mean of the simulated effect size.
    var : float
        Variance of the simulated effect size. Must be non-negative.

    Returns
    -------
    GRGCausalMutationModel
        Normal distribution Causal Mutation model.

    """

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        super().__init__("normal")

    def sim_effect_size(self, num_causal, rng):
        """
        This method simulates an effect size from a normal distribution.

        Parameters
        ----------
        num_causal : int
            Number of causal sites
        rng : numpy.random.Generator
            Random generator that will be used to simulate effect size.

        Returns
        -------
        float or array-like
            Simulated effect size of a causal mutation.
        """
        num_causal = self.check_parameter(num_causal, rng)
        beta = rng.normal(
            loc=self.mean,
            scale=np.sqrt(self.var),
            size=num_causal,
        )
        return beta


class GRGCausalMutationModelExponential(GRGCausalMutationModel):
    """
    Exponential distribution Causal Mutation model.

    Parameters
    ----------
    scale : float
        Scale of the exponential distribution. Must be non-negative.
    random_sign : bool, default False
        If True, effect sizes will randomly have either a positive or negative sign.
        If False, only positive values are simulated, reflecting the inherent properties
        of the exponential distribution.

    Returns
    -------
    GRGCausalMutationModel
        An object representing an exponential distribution model for Causal Mutations.

    """

    def __init__(self, scale, random_sign=False):
        self.scale = scale
        self.random_sign = random_sign
        super().__init__("exponential")

    def sim_effect_size(self, num_causal, rng):
        """
        Simulates an effect size from an exponential distribution.

        Parameters
        ----------
        num_causal : int
            Number of causal sites.
        rng : numpy.random.Generator
            Random generator used for simulation.

        Returns
        -------
        float or array-like
            Simulated effect size for each causal mutation.
        """
        num_causal = self.check_parameter(num_causal, rng)
        beta = rng.exponential(scale=self.scale, size=num_causal)
        if self.random_sign:
            beta = beta * rng.choice([-1, 1], size=num_causal)
        return beta


class GRGCausalMutationModelFixed(GRGCausalMutationModel):
    """
    Fixed value Causal Mutation model.

    Parameters
    ----------
    value : float
        Value of the simulated effect size.
    random_sign : bool, default False
        If True, :math:`1` or :math:`-1` will be randomly multiplied to the
        simulated effect sizes, such that we can simulate constant value effect
        sizes with randomly chosen signs.

    Returns
    -------
    GRGCausalMutationModel
        Fixed value Causal Mutation model.

    """

    def __init__(self, value, random_sign=False):
        self.value = value
        self.random_sign = random_sign
        super().__init__("fixed")

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a fixed Causal Mutation model.

        Parameters
        ----------
        num_causal : int
            Number of causal sites
        rng : numpy.random.Generator
            Random generator that will be used to simulate effect size.

        Returns
        -------
        float or array-like
            Simulated effect size of a causal mutation.
        """
        num_causal = self.check_parameter(num_causal, rng)
        beta = np.repeat(self.value, num_causal)
        if self.random_sign:
            beta = np.multiply(rng.choice([-1, 1], size=num_causal), beta)
        return beta


class GRGCausalMutationModelGamma(GRGCausalMutationModel):
    """
    Gamma distribution Causal Mutation model.

    Parameters
    ----------
    shape : float
        Shape of the gamma distribution. Must be non-negative.
    scale : float
        Scale of the gamma distribution. Must be non-negative.
    random_sign : bool, default False
        If True, :math:`1` or :math:`-1` will be randomly multiplied to the
        simulated effect sizes, such that we can simulate effect sizes with
        randomly chosen signs. If False, only positive values are being
        simulated as part of the property of the gamma distribution.

    Returns
    -------
    GRGCausalMutationModel
        Gamma distribution Causal Mutation model.
    """

    def __init__(self, shape, scale, random_sign=False):
        self.shape = shape
        self.scale = scale
        self.random_sign = random_sign
        super().__init__("gamma")

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a gamma distribution.

        Parameters
        ----------
        num_causal : int
            Number of causal sites
        rng : numpy.random.Generator
            Random generator that will be used to simulate effect size.

        Returns
        -------
        float or array-like
            Simulated effect size of a causal mutation.
        """
        num_causal = self.check_parameter(num_causal, rng)
        beta = rng.gamma(self.shape, self.scale, size=num_causal)
        if self.random_sign:
            beta = np.multiply(rng.choice([-1, 1], size=num_causal), beta)
        return beta


class GRGCausalMutationModelT(GRGCausalMutationModel):
    """
    Student's t distribution Causal Mutation model.

    Parameters
    ----------
    mean : float
        Mean of the simulated effect size.
    var : float
        Variance of the simulated effect size. Must be > 0.
    df : float
        Degrees of freedom. Must be > 0.

    Returns
    -------
    GRGCausalMutationModel
        Student's t distribution Causal Mutation model.
    """

    def __init__(self, mean, var, df):
        self.mean = mean
        self.var = var
        self.df = df
        super().__init__("t")

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a Student's t distribution.

        Parameters
        ----------
        num_causal : int
            Number of causal sites
        rng : numpy.random.Generator
            Random generator that will be used to simulate effect size.

        Returns
        -------
        float or array-like
            Simulated effect size of a causal mutation.
        """
        num_causal = self.check_parameter(num_causal, rng)
        beta = rng.standard_t(self.df, size=num_causal)
        beta = beta * np.sqrt(self.var) + self.mean
        return beta


class GRGCausalMutationModelMultivariateNormal(GRGCausalMutationModel):
    """
    Multivariate normal distribution Causal Mutation model.

    Parameters
    ----------
    mean : 1-D array_like, of length N
        Mean vector.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix. Must be symmetric and positive-semidefinite.

    Returns
    -------
    GRGCausalMutationModel
        Multivariate normal distribution Causal Mutation model.
    """

    def __init__(self, mean, cov):
        super().__init__("multivariate normal", num_trait=len(mean))
        self.mean = mean
        self.cov = cov

    def sim_effect_size(self, num_causal, rng):
        """
        This method returns an effect size from a multivariate normal distribution.

        Parameters
        ----------
        num_causal : int
            Number of causal sites
        rng : numpy.random.Generator
            Random generator that will be used to simulate effect size.

        Returns
        -------
        float or array-like
            Simulated effect size of a causal mutation.
        """

        num_causal = self.check_parameter(num_causal, rng)
        beta = rng.multivariate_normal(self.mean, self.cov, size=num_causal)
        return beta


class GRGCausalMutationModelMultivariateExponential(GRGCausalMutationModel):
    """
    Multivariate exponential distribution Causal Mutation model.

    Parameters
    ----------
    scales : float (vector of floats)
        Scales of the exponential distribution. Must be non-negative. One scale per
        Causal Mutation expected.
    random_sign : bool, default False
        If True, effect sizes will randomly have either a positive or negative sign.
        If False, only positive values are simulated, reflecting the inherent properties
        of the exponential distribution.

    Returns
    -------
    GRGCausalMutationModel
        Multivariate exponential distribution Causal Mutation model.
    """

    def __init__(self, scales, random_sign=False):
        super().__init__("multivariate exponential", num_trait=len(scales))
        self.scales = np.array(scales)
        self.random_sign = random_sign

    def sim_effect_size(self, num_causal, rng):
        beta = np.array(
            [rng.exponential(scale=scale, size=num_causal) for scale in self.scales]
        ).T
        if self.random_sign:
            signs = rng.choice([-1, 1], size=(num_causal, len(self.scales)))
            beta *= signs
        return beta


class GRGCausalMutationModelMultivariateFixed(GRGCausalMutationModel):
    """
    Multivariate fixed distribution Causal Mutation model.

    Parameters
    ----------
    values : float (vector of floats)
        Values of the simulated effect sizes.
    random_sign : bool, default False
        If True, :math:`1` or :math:`-1` will be randomly multiplied to the
        simulated effect sizes, such that we can simulate constant value effect
        sizes with randomly chosen signs.

    Returns
    -------
    GRGCausalMutationModel
        Multivariate fixed distribution Causal Mutation model.
    """

    def __init__(self, values, random_sign=False):
        super().__init__("multivariate fixed", num_trait=len(values))
        self.values = np.array(values)
        self.random_sign = random_sign

    def sim_effect_size(self, num_causal, rng):
        beta = np.tile(self.values, (num_causal, 1))
        if self.random_sign:
            signs = rng.choice([-1, 1], size=(num_causal, len(self.values)))
            beta *= signs
        return beta


class GRGCausalMutationModelMultivariateGamma(GRGCausalMutationModel):
    """
    Multivariate gamma distribution Causal Mutation model.

    Parameters
    ----------
    shapes : float (vector of floats)
        Shape of the gamma distribution. Must be non-negative.
    scales : float (vector of floats)
        Scale of the gamma distribution. Must be non-negative.
    random_sign : bool, default False
        If True, :math:`1` or :math:`-1` will be randomly multiplied to the
        simulated effect sizes, such that we can simulate effect sizes with
        randomly chosen signs. If False, only positive values are being
        simulated as part of the property of the gamma distribution.

    Returns
    -------
    GRGCausalMutationModel
        Multivariate gamma distribution Causal Mutation model.
    """

    def __init__(self, shapes, scales, random_sign=False):
        super().__init__("multivariate gamma", num_trait=len(shapes))
        self.shapes = np.array(shapes)
        self.scales = np.array(scales)
        self.random_sign = random_sign

    def sim_effect_size(self, num_causal, rng):
        beta = np.array(
            [
                rng.gamma(shape, scale, size=num_causal)
                for shape, scale in zip(self.shapes, self.scales)
            ]
        ).T
        if self.random_sign:
            signs = rng.choice([-1, 1], size=(num_causal, len(self.scales)))
            beta *= signs
        return beta


grg_causal_mutation_model_dict = {
    "normal": GRGCausalMutationModelNormal,
    "exponential": GRGCausalMutationModelExponential,
    "fixed": GRGCausalMutationModelFixed,
    "gamma": GRGCausalMutationModelGamma,
    "t": GRGCausalMutationModelT,
    "multivariate normal": GRGCausalMutationModelMultivariateNormal,
    "multivariate exponential": GRGCausalMutationModelMultivariateExponential,
    "multivariate fixed": GRGCausalMutationModelMultivariateFixed,
    "multivariate gamma": GRGCausalMutationModelMultivariateGamma,
}


def grg_causal_mutation_model(dist_type, **kwargs):
    """
    Return a Causal Mutation model corresponding to the specified model.

    Parameters
    ----------
    dist_type : str
        String describing the Causal Mutation model. The list of supported distributions
        are:
        * "normal": Normal distribution
        * "t": t distribution
        * "fixed": Fixed value
        * "exponential": Exponential distribution
        * "gamma": Gamma distribution
        * "multivariate normal": Multivariate normal distribution
        * "multivariate exponential": Multivariate exponential distribution
        * "multivariate fixed": Multivariate fixed distribution
        * "multivariate gamma": Multivariate gamma distribution


    **kwargs
        These parameters are used to specify the Causal Mutation model.

    Returns
    -------
    GRGCausalMutationModel
        Causal Mutation model that specifies the distribution of effect size simulation.

    Example uses for the different distribution types
    --------
    Normal distribution
    >>> model = grg_causal_mutation_model("normal", mean=0, var=1)

    T-distribution

    >>> model = grg_causal_mutation_model("t", mean=0, var=1, df=1)

    Fixed model distribution

    >>> model = grg_causal_mutation_model("fixed", value=1)

    Exponential distribution

    >>> model = grg_causal_mutation_model("exponential", scale=1)

    Exponential distribution (with negative values enabled)

    >>> model = grg_causal_mutation_model("exponential", scale=1, random_sign=True)

    Gamma distribution

    >>> model = grg_causal_mutation_model("gamma", shape=1, scale=2)

    Gamma distribution (with negative values enabled)

    >>> model = grg_causal_mutation_model("gamma", shape=1, scale=2, random_sign=True)

    Multivariate normal distribution

    >>> model = grg_causal_mutation_model("multivariate normal", mean=sample_mean, cov=sample_covariance)

    Usage for the other multivariate models is similar to the normal distribution. See specifications for further details.
    """
    dist = check_type(dist_type, "dist", str)
    model = dist.lower()
    if model not in grg_causal_mutation_model_dict:
        raise ValueError(
            "The given distribution '{}' is unknown. Choose from the options given {}".format(
                dist, sorted(grg_causal_mutation_model_dict.keys())
            )
        )
    current_model = grg_causal_mutation_model_dict[model](**kwargs)

    return current_model
