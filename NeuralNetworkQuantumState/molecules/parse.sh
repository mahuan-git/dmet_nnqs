#!/bin/bash

# dfile=c/log
dfile=$1
#echo $dfile
declare -A res_dict
# res_dict=([HF]=0 [CCSD]=0 [FCI]=0 [nqubit]=0 [nelec]=0)
res_dict[dist]=`grep dist $dfile | awk '{print $2}'`
res_dict[HF]=`grep Hartree $dfile | awk '{print $3}'`
res_dict[CCSD]=`grep "CCSD " $dfile | awk '{print $3}'`
res_dict[FCI]=`grep FCI $dfile | awk '{print $3}'`
res_dict[nqubit]=`grep qubits $dfile | awk '{print $4}'`
res_dict[nelec]=`grep electrons $dfile | awk '{print $4}'`
echo ${res_dict[dist]} ${res_dict[nqubit]} ${res_dict[nelec]} ${res_dict[HF]} ${res_dict[FCI]} ${res_dict[CCSD]} 
# echo ${res_dict[*]}

