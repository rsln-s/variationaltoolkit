# variationaltoolkit -- A Toolkit For Near-Term Quantum  

This is a set of tools that wrap up variational forms for optimization and more.

## Installation

```
conda create --name quantum intelpython3_core python=3.6
conda activate quantum
conda install numpy scipy cython ipython
pip install qiskit
git clone git@github.com:rsln-s/variationaltoolkit.git
cd variationaltoolkit
pip install -e .
```


## Optional Requirements

mpsbackend

### Using with Jupyter notebooks

```
conda install jupyter
python -m ipykernel install --user --name variationaltoolkit --display-name "variationaltoolkit"
```

from https://www.palmetto.clemson.edu/palmetto/jupyterhub_add_kernel.html

## Testing

We use Python unittest for tests. To run tests, run `python -m unittest` in the project root.

In general, if you are adding a test for method `foo` implemented in `variationaltoolkit/bar/blah.py`, the test should be in `test/bar/test_blah.py` and have name `test_foo`. Consult existing tests for examples.
