#!/bin/bash

#Usage: source setup.sh

cur_path=`pwd`
export PYTHONPATH=$cur_path:$cur_path/local_energy/:$PYTHONPATH
export LD_LIBRARY_PATH=$cur_path/local_energy/:$LD_LIBRARY_PATH
jobpath=`realpath test/job_script/showjob.sh`
alias sq="sh ${jobpath}"
alias k9="kill -9"
