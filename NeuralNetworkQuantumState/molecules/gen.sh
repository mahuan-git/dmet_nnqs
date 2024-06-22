#!/bin/bash
set -x

# sh gen.sh 140 1.40
#target=n2_$1
target=$1
#target=h2_$1
#target=lih_$1
target=$1_$2
mkdir $target
cp template/gen_ham.py $target 
cd $target
python=python3
$python gen_ham.py $2 | tee log
sh ../../parse.sh log
