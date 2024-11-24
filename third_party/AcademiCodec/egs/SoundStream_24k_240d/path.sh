#!/bin/bash
export MAIN_ROOT=`realpath ${PWD}/../../`

export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}
MODEL=encodec
export BIN_DIR=${MAIN_ROOT}/academicodec/models/${MODEL}