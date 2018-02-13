#!/bin/bash

source scripts/includes.sh
TEXT_RED=`tput setaf 1`
TEXT_GREEN=`tput setaf 2`
TEXT_RESET=`tput sgr0`
CONFIG_FILE=scripts/nvvp.cfg
NET_DIR=$1
NETWORK_NAME=$(basename $NET_DIR)
RESULT_DIR="$NETWORK_DIR/data/"
OUTPUT=$(file_name $NET_DIR 'CUDA' 0)
NVPROF_OPTIONS="-o ${OUTPUT}.nvvp --cpu-profiling on --normalized-time-unit s --unified-memory-profiling off"
NVPROF_OPTIONS="--dependency-analysis  $NVPROF_OPTIONS"
#NVPROF_OPTIONS="--analysis-metrics $NVPROF_OPTIONS"

if [ ! -f $CONFIG_FILE ];
then
	echo "[ERROR] Profile config file ${TEXT_RED}nvvp.cfg${TEXT_RESET} does not exist."
	exit 1
fi

if (($# < 1))
then
    echo "how to use: $0 network_dir"
    exit 1
fi

SOLVER=$NET_DIR/solver.prototxt 

COMPUTE_PROFILE=1 COMPUTE_PROFILE_CONFIG=${CONFIG_FILE} nvprof ${NVPROF_OPTIONS} ./skepu_ann $SOLVER cuda

if [ $? -eq 0 ];
then
	echo "[INFO] ${TEXT_GREEN}Profiler Successed.${TEXT_RESET}"
    exit 0
else
	echo "[ERROR] ${TEXT_RED}Usage: $0 ./skepu_ann network_dir${TEXT_RESET}"
    rm $output
	exit 1
fi
