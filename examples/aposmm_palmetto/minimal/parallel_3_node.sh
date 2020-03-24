#!/bin/bash

# Runs copies of script one per node, giving it all 16 cores on the node

module add gnu-parallel_201903 

parallel \
   --sshloginfile "$PBS_NODEFILE" \
   --jobs 1 \
   --linebuffer \
    """
    source ~/bash_setup/packages_qcommunity_ml.sh
    export PATH="/home/rshaydu/soft/anaconda3/bin:$PATH"
    source activate qcommunity_ml

    export OMP_NUM_THREADS=1

    cd /home/rshaydu/quantum/variationaltoolkit/examples/aposmm_palmetto/minimal

    mpirun -np 16 --ppn 16 python aposmm_complete_graph.py -n {1}
    """ :::  $(seq 10 12)
