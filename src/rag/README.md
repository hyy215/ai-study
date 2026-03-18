
# RAG (Retrieval-Augmented Generation) 学习模块

该目录包含了关于 RAG（检索增强生成）技术的核心概念实现与学习示例。目前主要聚焦于向量嵌入（Vector Embeddings）、相似度检索与可视化。

> **参考来源**：本章节内容基于 DeepLearning.AI 的课程 [Retrieval Augmented Generation (RAG)
](https://learn.deeplearning.ai/courses/retrieval-augmented-generation/information)。我们将 Python Notebook 中的核心概念迁移到了 TypeScript 环境中实现。

## 目录结构

```
src/rag/
├── vector-embeddings/       # 向量嵌入与检索核心模块
│   ├── demo.ts              # 综合演示脚本（包含检索与可视化）
│   ├── retrieval.ts         # 核心检索逻辑（模型加载、向量生成、文档排序）
│   ├── vectorUtils.ts       # 数学工具库（余弦相似度、欧氏距离等）
│   ├── pcaVisualization.ts  # PCA 降维可视化工具
│   ├── viz.html             # 可视化结果展示页面（ECharts）
│   └── viz-data.js          # 自动生成的可视化数据文件
└── retrieval-metrics/       # 检索评估模块
    ├── evaluate.ts          # 评估执行脚本（包含模拟数据集）
    └── metrics.ts           # 核心指标实现（Precision@K, Recall@K）
└── weaviate/                # Weaviate 向量数据库集成示例
    ├── inference-server.ts  # 本地推理服务（提供向量化与重排 API）
    ├── weaviate-main.ts     # Weaviate 核心逻辑（Schema、导入、搜索）
    └── mock-data.ts         # 测试数据
```

## 核心功能

### 1. 向量生成与检索 (`vector-embeddings/retrieval.ts`)
- 使用 `@huggingface/transformers` 加载预训练模型。
- 当前默认模型：`Xenova/paraphrase-multilingual-MiniLM-L12-v2`（支持多语言，包括中文）。
- 提供两种相似度度量标准：
  - **余弦相似度 (Cosine Similarity)**：衡量向量方向的一致性，语义检索首选。
  - **欧几里得距离 (Euclidean Distance)**：衡量向量空间距离。

### 2. 向量可视化 (`vector-embeddings/pcaVisualization.ts` & `viz.html`)
- 使用 **PCA (主成分分析)** 算法将 384 维（或 768 维）的高维向量压缩到 2 维平面。
- 生成 ECharts 图表，直观展示查询（Query）与文档（Documents）在语义空间中的分布关系。

### 3. 检索评估 (`retrieval-metrics/evaluate.ts`)
- 实现了 **Precision@K (前K项准确率)** 和 **Recall@K (前K项召回率)** 指标。
- 使用一个包含 4 个类别的模拟中文数据集（太空、图形学、冰球、医学）进行自动化评估。
- 分析不同 K 值（如 1, 3, 5）下的检索性能。

### 4. 数学基础 (`vector-embeddings/vectorUtils.ts`)
- 实现了点积（Dot Product）、L2 范数（Norm）、余弦相似度和欧氏距离的底层算法。

### 5. Weaviate 向量数据库集成 (`weaviate/weaviate-main.ts`)
- **嵌入式 Weaviate (Embedded Weaviate)**：在 Node.js 中直接启动 Weaviate 实例，无需 Docker。
- **自定义推理服务 (`weaviate/inference-server.ts`)**：
  - 使用 `@huggingface/transformers` 搭建本地 API。
  - 模拟 Weaviate 的 `text2vec-transformers` 和 `reranker-transformers` 模块接口。
- **高级搜索功能展示**：
  - **标量筛选 (Scalar Filtering)**：类似于 SQL 的 `WHERE` 子句。
  - **BM25 关键字搜索**：传统全文检索。
  - **混合搜索 (Hybrid Search)**：结合向量语义搜索与 BM25 关键字搜索，并支持 alpha 参数调节权重。
  - **重排序 (Reranking)**：使用 Cross-Encoder 模型对向量检索结果进行精细排序，提升准确性。

## 快速开始

### 1. 安装依赖
确保你已经安装了项目依赖：
```bash
pnpm install
```

### 2. 运行检索演示
运行 `demo.ts` 脚本，它会执行以下操作：
1. 下载并加载嵌入模型。
2. 对一组中文旅游景点描述进行向量化。
3. 执行查询："推荐一些亚洲值得一去的旅游景点"。
4. 打印检索排名结果。
5. 生成可视化数据。

```bash
npx tsx src/rag/vector-embeddings/demo.ts
```

### 3. 查看可视化结果
运行 demo 后，会在同目录下生成 `viz-data.js`。
在浏览器中打开 `src/rag/vector-embeddings/viz.html` 即可查看散点图。

### 4. 运行评估脚本
测试检索系统在多分类任务上的性能：

```bash
npx tsx src/rag/retrieval-metrics/evaluate.ts
```

它会输出每个查询在不同 K 值下的 Precision 和 Recall，以及整体平均指标。

### 5. 运行 Weaviate RAG 演示
该演示包含两个部分：本地推理服务和 Weaviate 主程序。

1. **启动推理服务**（在一个终端窗口中运行）：
   ```bash
   npx tsx src/rag/weaviate/inference-server.ts
   ```
   等待显示 "🚀 推理服务器运行在 http://127.0.0.1:8080"。

2. **运行 Weaviate 主程序**（在另一个终端窗口中运行）：
   ```bash
   npx tsx src/rag/weaviate/weaviate-main.ts
   ```
   该脚本将演示数据导入、标量过滤、混合搜索以及重排序等高级功能。

## 学习笔记

### 为什么检索结果可能不完美？
在演示中，你可能会发现即使查询包含“亚洲”，模型也可能推荐“班夫国家公园”（加拿大）。这是因为：
1. **模型局限性**：小型模型更关注语义相似度（如“旅游”、“绝佳去处”），而对地理实体的排他性理解较弱。
2. **缺乏显式过滤**：纯向量检索基于语义距离。解决办法通常结合 **元数据过滤 (Metadata Filtering)** 或 **混合检索 (Hybrid Search)**。
