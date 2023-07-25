from tqdm import tqdm

from deepom.aligner import DeepOMAligner
from deepom.localizer import DeepOMLocalizer
from utils.alignment import AlignmentAnalysis
from utils.bnx_images import ImageReader
from utils.bnx_parse import BNXParser
from utils.env import ENV
from utils.genome import GenomePatternMapper

bnx_parser = BNXParser()
deep_om_localizer = DeepOMLocalizer()
genome_pattern_mapper = GenomePatternMapper(
    pattern='CTTAAG',
    record_id_prefix='NC_',
    # record_id_prefix='NC_000001.11'
)

with open('./data/GCF_000001405.40_GRCh38.p14_genomic.fna') as f:
    references = {
        record['id']: genome_pattern_mapper.find_pattern(record)['positions']
        for record in tqdm(genome_pattern_mapper.sequence_records(f))
    }

aligner = DeepOMAligner(references=references, scale=335)

bnx_file = './data/T1_chip2_channels_swapped.bnx'

with open(bnx_file) as f:
    image_reader = ImageReader(runs=bnx_parser.parse_runs(f), bionano_images_dir="./data/bionano_runs/2022-04-19")

with open(bnx_file) as f:
    for molecule in bnx_parser.iter_molecules(f):
        bnx_molecule = BNXParser.BNXMolecule(**molecule)
        # skip molecules that span multiple FOVs, since image extraction doesn't work well for them yet
        if bnx_molecule.StartFOV != bnx_molecule.EndFOV:
            continue

        print(molecule['MoleculeID'])

        image = image_reader.read_image(molecule=molecule)['image']

        if image is not None:
            locs = deep_om_localizer.compute_localizations(image=image)['localizations']
            alignment = aligner.align(localizations=locs)
            analysis = AlignmentAnalysis(**alignment, references=references)

            print(repr(analysis))
