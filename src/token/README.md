# Token Colorizer

This tool visualizes how a Transformer model (specifically GPT-2 by default) tokenizes text. It assigns different background colors to each token so you can clearly see the boundaries between them.

## Prerequisites

Ensure you have installed the project dependencies:

```bash
pnpm install
```

## How to Run

You can run this tool using `tsx` (TypeScript Executor):

```bash
npx tsx src/token/tokenColorizer.ts
```

## Usage

1. Run the command above.
2. Wait for the tokenizer model (`Xenova/gpt2`) to download/load (first time might take a few seconds).
3. Type any sentence at the `Input >` prompt and press Enter.
4. View the colorized output and the list of tokens with their IDs.
5. Type `exit` to quit.

## Example

**Input:**
```
Hello world, this is a test.
```

**Output:**
The tool will display "Hello", " world", ",", " this", " is", " a", " test", "." with alternating background colors.

## Technologies

- **[@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers)**: For running the tokenizer in Node.js.
- **[chalk](https://www.npmjs.com/package/chalk)**: For terminal styling and coloring.
