from pathlib import Path

import numpy


class Config:
    HOME_DIR = Path.home()
    PROJECT_NAME = "OM"
    OUT_DIR = HOME_DIR / "out"
    DATA_DIR = HOME_DIR / "data"

    REFALIGNER = HOME_DIR / "bionano_sw/tools/pipeline/Solve3.7_03302022_283/RefAligner/1.0/RefAligner"
    BIONANO_RUN_DIR = DATA_DIR / "bionano_runs/2022-04-19"
    REF_CMAP_FILE = BIONANO_RUN_DIR / "hg38_DLE1_0kb_0labels.cmap"
    BNXDB_FILE = HOME_DIR / "bnx.db" #we should copy this, mnt, to file server
    BNX_FILE = BIONANO_RUN_DIR / "T1_chip2_channels_swapped.bnx"
    XMAP_FILE = BIONANO_RUN_DIR / "exp_refineFinal1.xmap"
    BIONANO_IMAGES_DIR = BIONANO_RUN_DIR
    
    GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
    EVALUATION_DIR = OUT_DIR/ "eval_images"
    BIONANO_BNX_SCALE = 375
    BIONANO_NOMINAL_SCALE = 335
    NM_PER_BP_NOMINAL = 0.34

    MAX_RETRIES = 3
    WANDB_FIGURE = "wandb_figure"
    WANDB_PREFIX = "wandb_"
    TEST_DIR = "test"
    LOCALIZER_TRAINING_OUTPUT_DIR = OUT_DIR / "LocalizerModule"
    CHECKPOINT_SEARCH_DIR = Path(__file__).parent.parent / "data/pretrained"
    CHECKPOINT_FILE: str = "checkpoint.pickle"
    
    SECOND_CHECKPOINT_SEARCH_DIR = Path(__file__).parent.parent / "data2/pretrained"
    TEST_LIST_FILE = SECOND_CHECKPOINT_SEARCH_DIR / "test_molecule_list"

    MY_LOCALIZER_TRAINING_OUTPUT_DIR = OUT_DIR / "DataLocalizer"
    
class Consts:
    PI = numpy.pi
    _2PI = PI * 2
    PI_HALF = PI / 2
    SQRT_2PI = numpy.sqrt(_2PI)
    PI_QUARTER = PI / 4
