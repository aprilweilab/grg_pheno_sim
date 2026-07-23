from typing import Optional, List
import glob
import numpy
import os
import pygrgl
import shutil
import subprocess

try:
    import pygrgl_spmv as _pygrgl_spmv
except ImportError:
    _pygrgl_spmv = None

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")
REPO_ROOT = os.path.join(THIS_DIR, "..")

TEST_GRG = os.path.join(REPO_ROOT, "demos", "data", "test-200-samples.vcf.gz")


def construct_grg(
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    jobs: int = 4,
    is_test_input: bool = True,
    ignore_missing: bool = False,
) -> str:
    if input_file is None:
        input_file = TEST_GRG
        is_test_input = False
    cmd = [
        "grg",
        "construct",
        "--force",
        "-p",
        "10",
        "-j",
        str(jobs),
        os.path.join(INPUT_DIR, input_file) if is_test_input else input_file,
    ]
    if ignore_missing:
        cmd.append("--ignore-missing")
    if output_file is not None:
        cmd.extend(["-o", output_file])
    else:
        output_file = os.path.basename(input_file) + ".final.grg"
    subprocess.check_call(cmd)
    return output_file


# Split a GRG and load all the parts, and returns them _sorted by position_
def split_and_load(
    grg_filename: str,
    out_dir: str,
    size_per: int,
    jobs: int,
    cleanup: bool = True,
    filenames: Optional[List] = None,
):
    subprocess.check_output(
        [
            "grg",
            "split",
            "-j",
            str(jobs),
            grg_filename,
            "-s",
            str(size_per),
            "-o",
            out_dir,
        ]
    )
    grgs = []
    for fn in glob.glob(os.path.join(out_dir, "*.grg")):
        grgs.append(pygrgl.load_immutable_grg(fn))
        if filenames is not None:
            filenames.append(fn)
    grgs.sort(key=lambda g: g.bp_range[0])
    if cleanup:
        shutil.rmtree(out_dir)
    return grgs


def make_grg_sparse_mat(
    positions: List[int],
    ref_alleles: List[str],
    alt_alleles: List[str],
    matrix: numpy.typing.NDArray,
    miss: Optional[numpy.typing.NDArray] = None,
    ploidy: int = 1,
) -> pygrgl.GRG:
    """
    Create a GRG that is equivalent to the sparse matrix representations of the given genotype
    matrix. I.e., this is not a GRG that compresses the data, but just a naive graph representation
    of the mapping between mutations and samples.
    """
    N = matrix.shape[0]
    M = len(positions)
    assert M == len(ref_alleles)
    assert M == len(alt_alleles)
    assert matrix.shape[1] == M
    g = pygrgl.MutableGRG(N, ploidy)

    def add_samples(sample_vector):
        # Create the mutation node
        node = g.make_node()
        for j in numpy.flatnonzero(sample_vector):
            g.connect(node, j)
        if ploidy == 2:
            dosage = sample_vector[0::ploidy] + sample_vector[1::ploidy]
            coals = numpy.count_nonzero(dosage == ploidy)
            g.set_num_individual_coals(node, coals)
        return node

    pos2miss = {}
    for i in range(M):
        sample_vector = matrix[:, i]
        mut_node = add_samples(sample_vector)
        pos = positions[i]
        miss_node = pygrgl.INVALID_NODE
        if pos in pos2miss:
            miss_node = pos2miss[pos]
        elif miss is not None:
            miss_vector = miss[:, i]
            if numpy.sum(miss_vector) > 0:
                miss_node = add_samples(miss_vector)
        pos2miss[pos] = miss_node
        g.add_mutation(
            pygrgl.Mutation(positions[i], alt_alleles[i], ref_alleles[i]),
            mut_node,
            miss_node,
        )
    return g
