## 硬件回归测试
### 步骤 1：生成参考数据
首先，在可信的环境中（例如，特定的硬件/编译器版本），使用 `graph_net.paddle.test_reference_device` 来生成基线日志和输出文件。
```bash
python -m graph_net.paddle.test_reference_device \
    --model-path /path/to/all_models/ \
    --reference-dir ./gold_reference \
    --compiler cinn \
    --device cuda

# --reference-dir: （必选）用于保存输出的 .log（性能/配置）文件和 .pdout（输出张量）文件的目录。
# --compiler: 指定编译器后端。
```
### 步骤 2：运行回归测试
更换硬件后，运行正确性测试脚本。此脚本会读取参考数据，使用完全相同的配置重新运行模型，并将新结果与“黄金”参考标准进行比对。
```bash
python -m graph_net.paddle.test_device_correctness \
    --reference-dir ./golden_reference \
    --device cuda
```
此脚本将报告任何失败情况（例如，编译错误、输出不匹配），并打印与参考日志的性能对比（加速/减速），从而帮助您快速识别性能回归问题。
