# DeepOM

DeepOM is a Python deep-learning software for optical mapping of DNA images to a genome reference.
See the [paper](https://doi.org/10.1093/bioinformatics/btad137) for details.

#   For inference only: Running the example script
Ensure you use python >= 3.10
```bash
cd tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python example.py
```


# Reproducing the figures from the paper

## For training: Installation

```shell
conda env create -f environment.yml
conda activate deepom
pip install -r requirements.txt
pip install -e .
```

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
