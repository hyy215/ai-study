import { AutoModelForCausalLM, AutoTokenizer, pipeline } from '@huggingface/transformers';
import chalk from 'chalk';

/**
 * Model Example CLI Tool
 * 
 * This script demonstrates the transformer decoder-only model architecture
 * and manual next-token prediction, inspired by Lesson 6 of the 
 * "How Transformer LLMs Work" course.
 */

async function main() {
    console.log(chalk.cyan.bold('\n--- Lesson 6: Model Example (TypeScript Version) ---\n'));

    // 1. Setup - Loading Model and Tokenizer
    // Using distilgpt2 as it's the most reliable small model for this environment
    const modelId = 'Xenova/distilgpt2';
    
    console.log(chalk.gray(`Loading model and tokenizer (${modelId})...`));
    const tokenizer = await AutoTokenizer.from_pretrained(modelId);
    const model = await AutoModelForCausalLM.from_pretrained(modelId, {
        dtype: 'fp32', 
    });
    console.log(chalk.green('✔ Model and Tokenizer loaded successfully!\n'));

    // 2. Generating a Text Response (Using Pipeline)
    console.log(chalk.blue.bold('Step 1: Generating text using a pipeline'));
    const generator = await pipeline('text-generation', modelId, {
        dtype: 'fp32',
    });
    
    const prompt = "France: Paris; Germany: Berlin; Italy: Rome; Spain:";
    console.log(chalk.white(`Prompt: "${prompt}"`));
    
    console.log(chalk.gray('Generating...'));
    const output = await generator(prompt, {
        max_new_tokens: 1,
        do_sample: false,
    });
    
    // @ts-ignore
    console.log(chalk.yellow(`Pipeline Output: "${output[0].generated_text}"\n`));

    // 3. Exploring Model Architecture (Conceptual)
    console.log(chalk.blue.bold('Step 2: Exploring Model Architecture'));
    // In Transformers.js, we can see some config info
    const config = model.config as any;
    console.log(chalk.white(`Vocabulary Size: ${config.vocab_size}`));
    console.log(chalk.white(`Hidden Size (Embedding Dim): ${config.n_embd || config.hidden_size}`));
    console.log(chalk.white(`Number of Layers: ${config.n_layer || config.num_hidden_layers}\n`));

    // 4. Manual Single Token Prediction
    console.log(chalk.blue.bold('Step 3: Manual Next-Token Prediction'));
    
    // a. Tokenize the input prompt
    console.log(chalk.gray('Tokenizing prompt...'));
    const inputs = await tokenizer(prompt);
    console.log(chalk.white(`Input IDs: [${inputs.input_ids.data}]`));
    
    // b. Model Inference
    console.log(chalk.gray('Running model inference...'));
    // Pass all inputs (input_ids, attention_mask, etc.) to the model
    const result = await model(inputs);
    const logits = result.logits; // Shape: [batch, sequence_length, vocab_size]
    
    console.log(chalk.white(`Logits Shape: [${logits.dims}]`));

    // c. Get the last token's logits
    // logits.data is a Float32Array
    // We want the last vector in the sequence dimension
    const [, seqLen, vocabSize] = logits.dims;
    const lastTokenLogits = logits.data.slice((seqLen - 1) * vocabSize, seqLen * vocabSize);
    
    // d. Find the token ID with the highest probability (Argmax)
    let maxVal = -Infinity;
    let tokenId = -1;
    for (let i = 0; i < lastTokenLogits.length; i++) {
        if (lastTokenLogits[i] > maxVal) {
            maxVal = lastTokenLogits[i];
            tokenId = i;
        }
    }
    
    console.log(chalk.white(`Predicted Token ID: ${tokenId}`));

    // e. Decode the predicted token ID
    const predictedToken = tokenizer.decode([tokenId]);
    console.log(chalk.green.bold(`\nPredicted Next Token: "${predictedToken}"`));
    
    console.log(chalk.cyan.bold('\n--- End of Lesson 6 Example ---\n'));
}

main().catch(err => {
    console.error(chalk.red('An error occurred:'), err);
});
