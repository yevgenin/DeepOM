# noinspection PyUnresolvedReferences
from dataclasses import dataclass, replace, field
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import fire
import imagecodecs
import numba
import numpy
import pandas
from matplotlib import pyplot
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DataFrame
from tqdm import tqdm

from om_decoder.utils import cached_func, obj_to_yaml, pyplot_show_window, extract_segment_from_endpoints


def read_jxr(file):
    return imagecodecs.jpegxr_decode(Path(file).read_bytes())

def read_jxr_segment(file, endpoints, segment_width):
    return extract_segment_from_endpoints(read_jxr(file)[None], endpoints=endpoints,
                                                            segment_width=segment_width)

def labeled_human_chromosomes(labeled_subseq, chrs=None):
    genome = cached_func(genome_human)()
    from om_decoder.genome_simulator import LabeledGenome

    items = [
        LabeledGenome(
            labeled_subseq=labeled_subseq,
            genome_seq=genome_seq
        ).make_labeled_mask()
        for genome_seq in tqdm(genome.genome_seqs, desc='genome seqs', disable=False)
    ]
    if chrs is not None:
        return [items[i - 1] for i in chrs]
    else:
        return items


def genome_human(seq_len_limit=None, name_filter='CM.*'):
    from om_decoder.data_config import OrganismsConfig
    from om_decoder.genome_reader import OrganismGenome, NCBI
    organism = OrganismsConfig.Homo_sapiens
    return OrganismGenome(
        seq_per_organism_limit=None,
        name_filter=name_filter,
        seq_len_limit=seq_len_limit,
        organism=organism,
        assembly_id=organism.ncbi_refseq_assembly_id,
        ncbi_table=cached_func(read_ncbi_table)(url=NCBI.ASSEMBLY_SUMMARY_GENBANK_URL),
    ).read_seqs()


def genome_ecoli(seq_len_limit=None, name_filter=None):
    from om_decoder.data_config import OrganismsConfig
    from om_decoder.genome_reader import OrganismGenome, NCBI
    organism = OrganismsConfig.Escherichia_coli_ATCC_700926
    return OrganismGenome(
        seq_per_organism_limit=None,
        name_filter=name_filter,
        seq_len_limit=seq_len_limit,
        organism=organism,
        assembly_id=organism.ncbi_refseq_assembly_id,
        ncbi_table=cached_func(read_ncbi_table)(url=NCBI.ASSEMBLY_SUMMARY_GENBANK_URL),
    ).read_seqs()


def genome_from_ncbi(ncbi_refseq_assembly_id: str, seq_len_limit=None, name_filter=None):
    from om_decoder.genome_reader import OrganismGenome, NCBI
    return OrganismGenome(
        seq_per_organism_limit=None,
        name_filter=name_filter,
        seq_len_limit=seq_len_limit,
        assembly_id=ncbi_refseq_assembly_id,
        ncbi_table=cached_func(read_ncbi_table)(url=NCBI.ASSEMBLY_SUMMARY_GENBANK_URL),
    ).read_seqs()


def labeled_seq(genome_id, labeled_subseq, seq_len_limit=None, name_filter=None):
    from om_decoder.data_config import LabeledSubseq
    from om_decoder.genome_simulator import LabeledGenome

    return LabeledGenome(genome_seq=cached_func(genome_from_ncbi)(genome_id).genome_seqs[0],
                         labeled_subseq=labeled_subseq).make_labeled_mask()


def _test_genomes():
    genome = genome_ecoli()
    print(obj_to_yaml(genome))
    print(len(genome.genome_seqs[0].seq))


def subseq_match_mask(seq, subseq):
    mask = (sliding_window_view(seq, window_shape=len(subseq)) == subseq).all(axis=1)
    mask_locs, = numpy.nonzero(mask)
    return mask_locs


def read_ncbi_table(url) -> DataFrame:
    print(url)
    data = cached_func(url_read)(url)
    df = pandas.read_csv(BytesIO(data), sep='\t', skiprows=1)
    return df


def url_read(url):
    return urlopen(url).read()


def sparse_gaussian_filter(
        positions: ndarray,
        scale: float, num_eval: int = None, space_size: int = None,
        sigma: float = None,
        cutoff_sigmas: int = 5,
):
    if sigma is None:
        sigma = scale

    if num_eval is None:
        num_eval = space_size // scale

    assert num_eval > 0

    values = numpy.zeros(num_eval, dtype='float64')
    cutoff = cutoff_sigmas * sigma
    _filter(values, positions, scale, sigma, cutoff)

    return values


@numba.njit
def _filter(values, positions, scale, sigma, cutoff):
    for pos in positions:
        start, stop = pos - cutoff, pos + cutoff
        start, stop = int(start / scale), int(stop / scale)
        for i in range(start, stop):
            if 0 <= i < len(values):
                eval_pos = i * scale
                arg = (eval_pos - pos) / sigma
                value = numpy.exp(-0.5 * arg ** 2)
                values[i] += value


def _test_sparse_gaussian_filter():
    scale = 300
    n = 10 ** 5
    k = 4000
    pos = [10000, 10900]
    y = sparse_gaussian_filter(pos, space_size=n, scale=scale, sigma=scale)
    print(y)
    x = numpy.linspace(0, n, len(y))
    with pyplot_show_window():
        pyplot.plot(x, y)


if __name__ == '__main__':
    fire.Fire()
