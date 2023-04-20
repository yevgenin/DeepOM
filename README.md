# DeepOM

DeepOM is a Python deep-learning software for optical mapping of DNA images to a genome reference.
See the [paper](https://doi.org/10.1093/bioinformatics/btad137) for details.

# Installation

```shell
conda env create -f environment.yml
conda activate deepom
pip install -r requirements.txt
pip install -e .
```

# Getting started

-   Reproducing the figures from the paper: See Jupyter notebooks in the [figures](figures/) dir.
-   Running the localizer net training:
    ```shell
    conda activate deepom
    python deepom/localizer.py
    ```
-   Running the benchmark:
    ```
    conda activate deepom
    python deepom/bionano_compare.py
    ```
