def test_variance(phenotype_df):
    """
    Tests the variance condition in the phenotypic data.
    """
    var_pheno = phenotype_df["phenotype"].var()
    var_genetic = phenotype_df["genetic_value"].var()
    var_epsilon = phenotype_df["environmental_noise"].var()

    test = var_pheno - (var_genetic + var_epsilon)

    return test
