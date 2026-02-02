# Model Example (Lesson 6)

This tool demonstrates the core components of a Transformer decoder-only model, mirroring the "Model Example" lesson from the DeepLearning.AI course. It shows how a model processes text from raw input to next-token prediction.

## Features

- **Pipeline Generation**: Uses the high-level `pipeline` API to generate a text completion.
- **Architecture Exploration**: Displays key model parameters like vocabulary size, embedding dimension, and number of layers.
- **Manual Prediction**: Deconstructs the prediction process step-by-step:
  1. **Tokenization**: Converting text to numerical IDs.
  2. **Model Inference**: Running the transformer blocks to get logits.
  3. **Logits Extraction**: Accessing the prediction scores for the next token.
  4. **Argmax & Decoding**: Finding the most likely token ID and converting it back to text.

## How to Run

You can run this tool using `tsx`:

```bash
npx tsx src/model/modelExample.ts
```

## Technologies

- **[@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers)**: For running the Transformer model and tokenizer in Node.js.
- **[chalk](https://www.npmjs.com/package/chalk)**: For terminal styling.
