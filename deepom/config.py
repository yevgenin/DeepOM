from pathlib import Path


class Config:
    BIONANO_BNX_SCALE = 375
    BIONANO_NOMINAL_SCALE = 335
    NM_PER_BP_NOMINAL = 0.34
    MAX_RETRIES = 3
    WANDB_FIGURE = "wandb_figure"
    WANDB_PREFIX = "wandb_"
    TEST_DIR = "test"
    KW_LINE = dict(lw=1, ms=1)
    KW_IMSHOW = dict(cmap='gray', interpolation='none', aspect='auto')
    LOCVEC_OUT_DIR = "locvecs"
    SEGMENTS_OUT_DIR = "segments"
    IMAGE_DIR = "."
    NAPARI_KW = dict(blending="additive", opacity=.7)
    KW_EVENTPLOT = {"lineoffsets": 0, "linelengths": .1}
    PROFILE_WIDTH = 17
    ALIGN_MAP_PLOT_SHAPE = 1000
    PROJECT_NAME = "OM"
    OUT_DIR = "out"
    DATA_DIR = "data"
    HOME = Path.home()
    CHECKPOINT_FILE: str = "checkpoint.pickle"
    REFSEQ_T7 = "GCF_000844825.1"
    REFSEQ_LAMBDA = "GCF_000840245.1"
    REFSEQ_ECOLI = "GCF_000005845.2"
