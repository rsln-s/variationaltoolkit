#!/bin/bash

python -c "import libensemble"
if [ $? -eq 0 ]
then
    for f in aposmm/test_*.py;
    do
        mpirun -np 3 python $f
    done
else
    echo "Did not find package libensemble, ignoring aposmm tests"
fi
