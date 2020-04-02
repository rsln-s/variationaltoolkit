# variationaltoolkit -- A Toolkit For Near-Term Quantum  

This is a set of tools that wrap up variational forms for optimization and more.

## Package Overview
![Flowchart](/images/HighLevelVariationalToolkit.png)

See tests for examples of how to use each part of the package. An in-depth flowchart documenting each function and the interactions can be found in images/VariationalToolkit.png

The flowchart can be edited within draw.io. The source can be found at https://drive.google.com/file/d/16l2bXQnwfoDn0K8B0lD2cfmAXarktajm/view?usp=sharin

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

mpsbackend (private, access by request)

## Using with Jupyter notebooks

```
conda install jupyter
python -m ipykernel install --user --name variationaltoolkit --display-name "variationaltoolkit"
```

from https://www.palmetto.clemson.edu/palmetto/jupyterhub_add_kernel.html

## Testing

To run tests, run `test/run_tests.sh` in the project root. It runs both unittest for sequential tests and mpi tests for aposmm.

In general, if you are adding a test for method `foo` implemented in `variationaltoolkit/bar/blah.py`, the test should be in `test/bar/test_blah.py` and have name `test_foo`. Consult existing tests for examples.
