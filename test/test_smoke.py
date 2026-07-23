import os
import pandas
import pygrgl
import sys
import unittest
from grg_pheno_sim.phenotype import sim_phenotypes

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR)
from testing_utils import construct_grg

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg(
            output_file="test.smoke.grg", is_test_input=False
        )
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)
        assert (
            cls.grg.ploidy == 2
        ), "Phenotype simulation currently only supports diploid datasets"

    def test_simple_pheno(self):
        SEED = 1999
        H2 = 0.4
        expected = pandas.read_csv(os.path.join(INPUT_DIR, "smoke.baseline.csv"))

        phenotypes = sim_phenotypes(
            self.grg,
            num_causal=100,
            random_seed=SEED,
            normalize_phenotype=True,
            normalize_genetic_values_before_noise=True,
            heritability=H2,
            save_effect_output=False,
            header=True,
        )
        pandas.testing.assert_frame_equal(expected, phenotypes)

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
