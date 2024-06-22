#!/bin/bash

# Usage: source env.sh

cur_path=`pwd`
export PYTHONPATH=$cur_path:$PYTHONPATH
# export JULIA_LOAD_PATH=$cur_path/:$JULIA_LOAD_PATH