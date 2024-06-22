#!/bin/bash

if [ "$#" -ne 2 ]; then
    SOURCE_DIR="../outfile"
    TARGET_DIR="./result"
else
    SOURCE_DIR="$1"
    TARGET_DIR="$2"
fi

echo "Source Directory: $SOURCE_DIR"
echo "Target Directory: $TARGET_DIR"

# 源目录（包含要搜索的文件）
#SOURCE_DIR="../outfile"
#
## 目标目录（保存结果文件）
#TARGET_DIR="./result"

# 检查目标目录是否存在，如果不存在，则创建
mkdir -p $TARGET_DIR
KEY_WORD=eloc_mean

# 遍历源目录中的所有文件
for file in $SOURCE_DIR/*; do
    # 检查文件是否包含“EMEAN”
    if grep -q "${KEY_WORD}" "$file"; then
        new_name=$(basename $file | sed 's/^out\./result-/')
        grep "${KEY_WORD}" "$file" > "$TARGET_DIR/$new_name"
    fi
done
                                
