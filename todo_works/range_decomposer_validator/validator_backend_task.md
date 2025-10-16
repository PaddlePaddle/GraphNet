GraphNet&ai4c 第三轮开源任务
# AI4C**计算图分解验证器**
## 任务背景
编译机床建设的第一个任务是对 GraphNet 的计算图数据集（及其元信息）进行有效切分拆解，以便 Agentic AI 学习样本，从而进行IR迁移、相似模式优化等任务。

我们设想中的 AI4C 子图分解功能包含以下模块：

1. 计算图区间分解器，负责分解操作执行，需要包含分解区间配置
2. 计算图分解方案验证器，对拆分后的子图做有效性验证

基于以上设计，我们计划先完成【计算图区间分解方案验证器】。我们计划复用现有的test_compiler及后续的评估方法，只新增一个RangeDecomposerValidatorBackend，作为验证器核心。

## 任务描述
该任务的目标是在todo_works.range_decomposer_validator下，实现一个完善的range_decomposer_validator：

1. 该validator作为backend导入graph_net.torch.test_compiler，相应的配置已写入test_compiler代码；
2. 该validator接收一个【原模型】的torch.nn.Module，通过文件路径解析【分解后模型】的多个subgraph，重新输出一个组合后的torch.nn.Module；
3. 在解析过程中，默认【分解后模型】路径为【原模型】路径加上_decomposed，下有多个subgraph单独目录，例如/test/simple_CNN/的分解后模型包括/test/simple_CNN_decomposed/subgraph_0/.../test/simple_CNN_decomposed/subgraph_n/，每个subgraph的文件组成等同一份标准的GraphNet样本；
4. 在组合过程中，组合模型的forward是每个分解模型依次连接、嵌套而成，前一个模型的输出作为下一个模型的输入；
5. 该validator应当能够检测【原模型】与【分解后模型】的算子数量，去掉placeholder后进行比对，以验证其完整性；
6. 该validator应当根据【区间分解模式】配置文件（eg., 存放于/test/simple_CNN_decomposed/路径下，记录模型信息、子图数量、子图规模等，可参照graph_net.json设计），能够检测【分解后模型】是否按照预期的分解模式，以充分验证其完整性、分解验证过程前后一致性。

## 预期效果
根据 [https://github.com/PaddlePaddle/GraphNet?tab=readme-ov-file#%EF%B8%8F-compiler-evaluation](https://github.com/PaddlePaddle/GraphNet?tab=readme-ov-file#%EF%B8%8F-compiler-evaluation) 提供的评估方法，对于【单个模型计算图】存在以下测试步骤：

1. graph_net.torch.test_compiler，记录下原始log
2. graph_net.log2json，将log转化为JSON
3. graph_net.S_analysis，生成S和ES图象，并输出其各项参数

随后，根据ES图象t>0时的阶梯表现，我们可以分析出该样本的正确性。在当前GraphNet repo中，默认t=1的跳跃代表输出精度错误，t=3的跳跃代表编译运行等其它类别错误。

为了方便测试，我们在GraphNet/todo_works/range_decomposer_validator/test/下提供了简单的测试用例；开发者需要构造出更多测试用例，包含分解错误以及有placeholder的情形，以说明验证器达成了功能。

由于是单个样本测试，无需考虑性能提升，故预期使用所需的RangeDecomposerValidatorBackend后，对于正确拆分样本，ES图象应当是y=1的【一条直线】；对于错误或不完整的拆分样本，应当打印【错误报告】，或ES图象在t>0区域存在【阶梯状抬升】。

