#!/bin/bash

NUM_NODES=$1
NUM_PROCS=$2
COMM_TYPE="$3"
EXEC="$4"
PROB_SIZE=$5
NUM_ITERS=$6
LOCAL_SOLVE="$7"
PARTITION="$8"

NUM_RANKS_PER_PROC=1

EXEC_DIR=$PWD
EXEC_TYPE_DIR="${EXEC}"

mkdir -p $EXEC_TYPE_DIR

RESULTS_DIR=${EXEC_DIR}/${EXEC_TYPE_DIR}

cd $RESULTS_DIR

COMM_TYPE_FLAG="--enable_${COMM_TYPE}"
COMM_TYPE_DIR="${COMM_TYPE}"
OUT_DIR="$((${NUM_NODES}*${NUM_PROCS}))domains"
PROB_SIZE_DIR="${PROB_SIZE}local"
LOCAL_SOLVE_DIR="${LOCAL_SOLVE}"
PARTITION_DIR="${PARTITION}"
JOB_NAME="${COMM_TYPE}-${NUM_NODES}-${NUM_PROCS}"
TIME_FILE="subd"


mkdir -p $COMM_TYPE_DIR
cd $COMM_TYPE_DIR

mkdir -p $OUT_DIR
cd $OUT_DIR

mkdir -p $LOCAL_SOLVE_DIR
cd $LOCAL_SOLVE_DIR

mkdir -p $PARTITION_DIR
cd $PARTITION_DIR

mkdir -p $PROB_SIZE_DIR
cd $PROB_SIZE_DIR

export run_exec=$EXEC_DIR/benchmarking/bench_ras
export run_flags="--executor=${EXEC} --num_iters=${NUM_ITERS} --timings_file=${TIME_FILE} --partition=${PARTITION} --set_tol=1e-6 --explicit_laplacian --set_1d_laplacian_size=${PROB_SIZE} ${COMM_TYPE_FLAG} --local_solver=${LOCAL_SOLVE} --factor_ordering_natural --local_tol=1e-12 --enable_global_tree_check --enable_put_all_local_residual_norms --enable_global_check"

echo "#!/bin/bash -l" > job_script.sh
echo "#BSUB -o output-$JOB_NAME" >> job_script.sh
echo "#BSUB -e error-$JOB_NAME" >> job_script.sh
echo "#BSUB -J $JOB_NAME" >> job_script.sh
echo "#BSUB -nnodes $NUM_NODES" >> job_script.sh
echo "#BSUB -alloc_flags \"smt1\" " >> job_script.sh
echo "#BSUB -P GEN010" >> job_script.sh
echo "#BSUB -W 02:00" >> job_script.sh

NUM_RES_SETS="$((${NUM_NODES}*1))"
echo "date" >> job_script.sh
echo "export OMP_NUM_THREADS=1" >> job_script.sh
echo "export PAMI_DISABLE_IPC=1" >> job_script.sh
echo "jsrun --smpiargs=\"-gpu -async\" -n ${NUM_RES_SETS} -a ${NUM_PROCS} -g ${NUM_PROCS} -c ${NUM_PROCS} -r ${NUM_RANKS_PER_PROC} -d packed -b packed:1 --latency_priority gpu-cpu $run_exec $run_flags" >> job_script.sh
#echo "jsrun --smpiargs="-gpu" -n ${NUM_RES_SETS} -a ${NUM_PROCS} -g ${NUM_PROCS} -c ${NUM_PROCS} -r ${NUM_RANKS_PER_PROC} -d packed -b packed:1 --latency_priority gpu-cpu nvprof --analysis-metrics -fo prof2.nvvp $run_exec $run_flags" >> job_script.sh
bsub job_script.sh
