# Token 着色器

该工具可视化 Transformer 模型（默认特指 GPT-2）如何对文本进行 Token 化。它为每个 Token 分配不同的背景颜色，以便您可以清楚地看到它们之间的边界。

## 前提条件

确保您已安装项目依赖项：

```bash
pnpm install
```

## 如何运行

您可以使用 `tsx` (TypeScript Executor) 运行此工具：

```bash
npx tsx src/token/tokenColorizer.ts
```

## 使用方法

1. 运行上述命令。
2. 等待分词器模型 (`Xenova/gpt2`) 下载/加载（首次运行可能需要几秒钟）。
3. 在 `Input >` 提示符处输入任何句子并按回车。
4. 查看着色后的输出以及带有 ID 的 Token 列表。
5. 输入 `exit` 退出。

## 示例

**输入:**
```
Hello world, this is a test.
```

**输出:**
该工具将显示 "Hello", " world", ",", " this", " is", " a", " test", "." 并带有交替的背景颜色。

## 技术栈

- **[@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers)**: 用于在 Node.js 中运行分词器。
- **[chalk](https://www.npmjs.com/package/chalk)**: 用于终端样式美化和着色。
