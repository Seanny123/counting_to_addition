# Progressing from Counting to Addition
Code for the paper "[Improving with Practice: A Neural Model of Mathematical Development](http://compneuro.uwaterloo.ca/files/publications/aubin.2016.pdf)" and forthcoming eponymous paper in TopiCS. Presentation slides for the CogSci2016 talk can be [found here](https://1drv.ms/p/s!Auhg6REoCX4GgWpkZHvmFBIN4FWV).

## Requirements

Python is required to run these scripts. The scripts were written in Python 2, but should work in Python 3.
The Python packages required can be installed with `pip install -r requirements.txt`.

The specific version of the Nengo Python package was
[ceaf387](https://github.com/nengo/nengo/tree/ceaf387ada525c2f0e84ea91214d90cc99763d7c).

## Running the networks

- `with_feedback.py` is the final full-network version
- `counting_only.py` uses only the counting network
- `better_pred_run.py` generates the data from the prediction in the paper
- `hetmem_learning.py` and `autoens_learning.py` generate the data for the comparison of learning with and without Voja

## Analysis

All analysis was performed in Jupyter notebooks, which end with the file extension `ipynb`.
These can be opened by running `jupyter notebook BLAH.ipynb` where `BLAH.ipynb` is the name of the notebook you
desire to view.
