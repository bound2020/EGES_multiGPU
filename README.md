参考了两位大佬的代码，简单实现了论文《Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba》中的EGES，并可以运行在单机多GPU上。

训练pipeline：

1. 统计点击序列中的每个item pair的共现次数，保存为data/graph_node

2. 生成每个item的side info，保存为data/side_info_feature

3. 运行preprocess.py，根据图随机游走生成序列，保存为data/walk_seq

4. 运行eges_multigpu.py，生成item向量

   

- EGES部分代码参考：https://github.com/wangzhegeek/EGES
- 单机多卡部分代码参考：https://github.com/lomyal/simple-word-embedding