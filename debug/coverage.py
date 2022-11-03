from deepom.bionano_compare import BionanoCompare
from deepom.localizer import LocalizerModule

if __name__ == '__main__':
    compare = BionanoCompare()
    compare.data_prep.num_crops_per_size = 2
    compare.data_prep.num_sizes = 2
    compare.run_bionano_compare_a()
    compare = BionanoCompare()
    compare.data_prep.num_crops_per_size = 2
    compare.data_prep.num_sizes = 2
    compare.run_bionano_compare_b()
    next(LocalizerModule().task_steps())

