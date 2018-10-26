# IBM Quantum Experience backend for QCommunity

This is a backend for QCommunity -- quantum-accelerated framework for graph community detection 

## Installation

```
git clone git@github.com:rsln-s/ibmqxbackend.git
cd ibmqxbackend
pip install -e .
```

## Installing `local_qasm_simulator_cpp`

For the local high-performance simulator, follow instructions in qiskit-terra repo.

https://github.com/Qiskit/qiskit-terra/tree/master/src/qasm-simulator-cpp

For me, it was sufficient to remove stuff from CMakeLists.txt until it works, then
`mkdir out; cd out; cmake -DCMAKE_CXX_COMPILER=icpc ..; make`
