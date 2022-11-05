# DeepOM

DeepOM is a Python deep-learning software for optical mapping of DNA images to a genome reference.
See the [paper](https://www.biorxiv.org/content/10.1101/2022.11.04.512597v1) for details.

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
