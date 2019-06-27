#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=benchmark.out
#SBATCH --error=benchmark.error
#SBATCH --partition=bdwall
#SBATCH --account=STARTUP-RSHAYDULIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=bdwall
#SBATCH --time=24:00:00

# Setup My Environment
module load intel-parallel-studio/cluster.2016.4-egfblc6
export I_MPI_FABRICS=shm:tmi

export PATH="/home/rshaydulin/soft/anaconda3/bin:$PATH"
source activate qcommunity

cd /home/rshaydulin/dev/ibmqxbackend/ibmqxbackend/benchmarking

# Run My Program

for seed in {1..20};
do
    for t in  {1..36};
    do
        for q in {10..14};
        do
            /home/rshaydulin/soft/anaconda3/envs/qcommunity/bin/python benchmark_ansatz.py -q $q -t $t --save benchmark_bebop.csv
        done
    done
done
