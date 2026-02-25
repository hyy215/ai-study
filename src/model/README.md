# 模型示例

该工具演示了 Transformer decoder-only 模型的核心组件，对应 [DeepLearning.AI 课程](https://learn.deeplearning.ai/courses/how-transformer-llms-work/lesson/m3nid/model-example)中的“模型示例”课程。它展示了模型如何从原始输入文本处理到下一个 token 的预测。

## 功能特性

- **管道生成**: 使用高级 `pipeline` API 生成文本补全。
- **架构探索**: 显示关键模型参数，如词汇表大小、嵌入维度和层数。
- **手动预测**: 逐步解构预测过程：
  1. **Token 化**: 将文本转换为数字 ID。
  2. **模型推理**: 运行 transformer 块以获取 logits。
  3. **Logits 提取**: 获取下一个 token 的预测分数。
  4. **Argmax 与解码**: 找到可能性最大的 token ID 并将其转换回文本。

## 如何运行

您可以使用 `tsx` 运行此工具：

```bash
npx tsx src/model/modelExample.ts
```

## 技术栈

- **[@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers)**: 用于在 Node.js 中运行 Transformer 模型和分词器。
- **[chalk](https://www.npmjs.com/package/chalk)**: 用于终端样式美化。
