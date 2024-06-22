#!/usr/bin/env python
# encoding: utf-8

import os
import re

def extract_last_step(directory):
    # 获取目录下的所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    steps_dict = {}
    for file in files:
        filepath = os.path.join(directory, file)

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # 从后向前查找匹配的行
        for line in reversed(lines):
            match = re.search(r'==\[LOG\]== energy at the (\d+)-th step', line)
            if match:
                steps_dict[file] = int(match.group(1))
                break

    return steps_dict

# 调用函数
directory = "../outfile"
steps_data = extract_last_step(directory)
for filename, step in steps_data.items():
        print(f"{filename}: {step} steps")


