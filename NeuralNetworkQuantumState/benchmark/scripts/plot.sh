#!/bin/bash

# usage: sh plot.sh <input_data_dir> <output_result_data_dir>
# example: sh plot.sh nh3 result_nh3
#sh extract_emean.sh $1 $2
#python plot_all.py $2

#!/bin/bash

# 获取最后一个参数
last_arg="${@: -1}"

# 获取参数总数
num_args=$#

# 从第一个到倒数第二个参数，执行 sh extract_emean.sh
for (( i=1; i<$num_args; i++ )); do
    first_arg="${!i}"
    sh extract_emean.sh "$first_arg" "$last_arg"
done

python plot_all.py "$last_arg"
