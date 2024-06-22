> @谢岱佑 主要实现 @吴扬俊 修改整合

Benchmark包含30 qubit以内的17个分子的基准测试，主要用于正确性验证。

git commit id: 7ed09f35d6d3926420d7bffa33ff708f9e702fea
GPU: A100-40G
请确保版本上述一致，平台，配置文件相同，对程序精度无影响的改动情况下，程序执行结果应与benchmark保持一致。

目录结构：
- outfile: 原始的输出日志文件
- config: 配置文件（可用extract_config.sh从日志文件中提取）
- scripts：若干有用的脚本
    - extract_config.sh：从日志文件提取config文件
    - extract_emean.sh：从日志文件中提取能量，把outfile中的emean提取到result里
    - check_steps.py：计算每个体系跑到多少步了，可以一次性监测多个文件的进展
    - ref.fci：各个分子的FCI参考能量，跑别的体系可以把它的比较值也写进去
    - plot_all.py：绘图
    - plot.sh: 提取能量并绘图（一般将所有分子的输出文件放到一个目录，在sh plot.sh即可完成绘图）
