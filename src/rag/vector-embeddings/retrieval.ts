
import { pipeline, FeatureExtractionPipeline } from '@huggingface/transformers';
import { cosineSimilarity, euclideanDistance } from './vectorUtils.js';

// 初始化嵌入模型
// 'Xenova/paraphrase-multilingual-MiniLM-L12-v2' 体积更小且速度更快，适用于通用场景。
export const modelName = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';

/**
 * 获取特征提取器实例（单例模式，避免重复加载）
 */
let extractorInstance: FeatureExtractionPipeline | null = null;
export async function getExtractor(): Promise<FeatureExtractionPipeline> {
    if (!extractorInstance) {
        console.log(`正在加载模型: ${modelName}...`);
        extractorInstance = await pipeline('feature-extraction', modelName) as unknown as FeatureExtractionPipeline;
        console.log('模型加载完毕。');
    }
    return extractorInstance;
}

/**
 * 为单个字符串或字符串数组生成嵌入向量。
 */
export async function getEmbeddings(text: string | string[], extractor?: FeatureExtractionPipeline): Promise<number[][]> {
    const ext = extractor || await getExtractor();
    const output = await ext(text, { pooling: 'mean', normalize: true });
    
    // 输出是一个 Tensor。我们需要将其转换为普通的 JS 数组。
    // 使用特征提取 pipeline 时，输出通常是 [batch_size, hidden_size]
    const embeddings = output.tolist();
    return embeddings;
}

interface RetrieveResult {
    document: string;
    score: number;
    embedding?: number[]; // 可选：返回向量以便可视化
}

/**
 * 根据与给定查询的相似度（使用指定的度量标准）检索并排序文档。
 */
export async function retrieveRelevant(
    query: string, 
    documents: string[], 
    metric: 'cosine_similarity' | 'euclidean' = 'cosine_similarity'
): Promise<RetrieveResult[]> {
    const extractor = await getExtractor();

    console.log('正在生成向量...');
    // 为查询和文档生成嵌入向量
    const queryEmbedding = (await getEmbeddings(query, extractor))[0];
    const docEmbeddings = await getEmbeddings(documents, extractor);

    let vals: RetrieveResult[] = [];

    if (metric === 'cosine_similarity') {
        const scores = cosineSimilarity(queryEmbedding!, docEmbeddings);
        vals = documents.map((doc, i) => ({
            document: doc,
            score: scores[i]!,
            embedding: docEmbeddings[i]!
        }));
        // 按降序排序（越高越好）
        vals.sort((a, b) => b.score - a.score);
    } else if (metric === 'euclidean') {
        const distances = euclideanDistance(queryEmbedding!, docEmbeddings);
        vals = documents.map((doc, i) => ({
            document: doc,
            score: distances[i]!,
            embedding: docEmbeddings[i]!
        }));
        // 按升序排序（越低越好）
        vals.sort((a, b) => a.score - b.score);
    }

    return vals;
}
