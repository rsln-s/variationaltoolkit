#!/bin/bash

python -c "import libensemble"
if [ $? -eq 0 ]
then
    for f in aposmm/test_*.py;
    do
        mpirun -np 3 --ppn 3 python $f
    done
    ncores=`grep -c ^processor /proc/cpuinfo`
    if [ $ncores -ge 16 ]
    then
        mpirun -np 16 --ppn 16 python aposmm/16p_test_performance_aposmm_only.py
    fi
else
    echo "Did not find package libensemble, ignoring aposmm tests"
fi
