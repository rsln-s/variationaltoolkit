# IBM Quantum Experience backend for qcommunity_dwave

## Installation

```
git clone git@github.com:rsln-s/ibmqxbackend.git
cd ibmqxbackend
pip install -e .
```

## Installing `local_qasm_simulator_cpp`

https://github.com/Qiskit/qiskit-terra/tree/master/src/qasm-simulator-cpp

Remove stuff from CMakeLists.txt until it works
`mkdir out; cd out; cmake -DCMAKE_CXX_COMPILER=icpc ..; make`
