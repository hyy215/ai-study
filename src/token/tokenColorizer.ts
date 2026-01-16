import { AutoTokenizer } from '@huggingface/transformers';
import chalk from 'chalk';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { encoding_for_model, type TiktokenModel } from 'tiktoken';

/**
 * Tokenizer CLI Tool
 * 
 * This script allows choosing between @huggingface/transformers and tiktoken
 * to tokenize input text and visualizes the tokens with alternating colors.
 */

// Define a palette of colors to cycle through for visualization (Pastel/Light Theme)
const colors = [
    chalk.bgHex('#FFB3BA').black, // Pastel Pink
    chalk.bgHex('#FFDFBA').black, // Pastel Orange
    chalk.bgHex('#FFFFBA').black, // Pastel Yellow
    chalk.bgHex('#BAFFC9').black, // Pastel Green
    chalk.bgHex('#BAE1FF').black, // Pastel Blue
    chalk.bgHex('#E6B3FF').black, // Pastel Purple
];

interface TokenizerStrategy {
    name: string;
    init(): Promise<void>;
    tokenize(text: string): Promise<{ tokens: string[], ids: any[] }>;
}

class HuggingFaceTokenizer implements TokenizerStrategy {
    name = '@huggingface/transformers (Xenova/gpt2)';
    private tokenizer: any;

    async init() {
        console.log(chalk.gray('Loading tokenizer (Xenova/gpt2)...'));
        this.tokenizer = await AutoTokenizer.from_pretrained('Xenova/gpt2');
        console.log(chalk.green('✔ Tokenizer loaded successfully!'));
    }

    async tokenize(text: string) {
        const { input_ids } = await this.tokenizer(text);
        // Cast to any to handle different TypedArray return types
        const ids = Array.from(input_ids.data as any);
        
        const tokens: string[] = [];
        for (let i = 0; i < ids.length; i++) {
            const id = ids[i];
            const tokenStr = this.tokenizer.decode([id as any]);
            tokens.push(tokenStr);
        }
        return { tokens, ids };
    }
}

class TiktokenTokenizer implements TokenizerStrategy {
    name = 'tiktoken (gpt-4o)';
    private enc: any;

    async init() {
        // No async init needed for tiktoken really, but keeping interface consistent
        // We use gpt-4o as default for tiktoken example
        try {
            this.enc = encoding_for_model("gpt-4o");
            console.log(chalk.green('✔ Tiktoken encoder (gpt-4o) loaded successfully!'));
        } catch (e) {
             console.error(chalk.red('Failed to load tiktoken model'), e);
             throw e;
        }
    }

    async tokenize(text: string) {
        const encoded = this.enc.encode(text);
        // encoded is a Uint32Array (or similar), convert to regular array
        const ids = Array.from(encoded);
        
        const tokens: string[] = [];
        const decoder = new TextDecoder();

        for (const id of ids) {
             // tiktoken's decode returns Uint8Array
             const tokenBytes = this.enc.decode([id as number]); 
             // Convert Uint8Array to string
             const tokenStr = decoder.decode(tokenBytes);
             tokens.push(tokenStr);
        }
        
        return { tokens, ids };
    }
}

async function main() {
    const rl = readline.createInterface({ input, output });

    try {
        console.log(chalk.cyan('Select Tokenizer Library:'));
        console.log('1. @huggingface/transformers (GPT-2)');
        console.log('2. tiktoken (GPT-4o)');
        
        const choice = await rl.question(chalk.bold('Choice (1 or 2) > '));
        
        let strategy: TokenizerStrategy;
        
        if (choice.trim() === '2') {
            strategy = new TiktokenTokenizer();
        } else {
            strategy = new HuggingFaceTokenizer();
        }

        await strategy.init();

        console.log(chalk.cyan(`\nUsing ${strategy.name}`));
        console.log(chalk.cyan('Enter a sentence to see how it is tokenized. Type "exit" or press Ctrl+C to quit.\n'));

        while (true) {
            const answer = await rl.question(chalk.bold('Input > '));
            
            if (answer.trim().toLowerCase() === 'exit') {
                break;
            }

            if (!answer.trim()) {
                continue;
            }

            const { tokens, ids } = await strategy.tokenize(answer);

            let visualOutput = '';

            for (let i = 0; i < tokens.length; i++) {
                const tokenStr = tokens[i];
                // Apply color cycling
                const colorFn = colors[i % colors.length];
                if (colorFn) {
                    visualOutput += colorFn(tokenStr);
                }
            }

            console.log('\n' + chalk.bold('Tokens:'));
            console.log(visualOutput);
            
            console.log('\n' + chalk.bold('Token List:'));
            console.log(tokens.map((t, i) => `${i + 1}. ${JSON.stringify(t)} (ID: ${ids[i]})`).join('\n'));
            
            console.log(chalk.gray('\n--------------------------------------------------\n'));
        }

    } catch (error) {
        console.error(chalk.red('An error occurred:'), error);
    } finally {
        rl.close();
        process.exit(0);
    }
}

main();
