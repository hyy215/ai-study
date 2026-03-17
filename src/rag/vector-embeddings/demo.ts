
import { retrieveRelevant, getEmbeddings, getExtractor } from './retrieval.js';
import { performPCA } from './pcaVisualization.js';

(async () => {
    const documents = [
        "富士山是秋季探索的绝佳去处，景色令人叹为观止。",
        "圣托里尼在春季提供令人惊叹的美景供人欣赏。",
        "班夫国家公园是夏季游览如画风景的理想目的地。",
        "中国长城是冬季体验壮观景象的绝佳地点。",
        "挪威峡湾是春季乘船游览的神奇之地。",
        "布拉格是冬季漫步其中的迷人城市。",
        "京都的樱花在春季创造了美丽的景色。",
        "马拉喀什在秋季提供充满活力的市场和文化体验。",
        "马尔代夫是夏季享受的天堂般的度假胜地。",
        "维也纳的圣诞市场是冬季探索的节日乐趣。"
    ];

    const query = "推荐一些亚洲值得一去的旅游景点。";

    console.log(`查询: "${query}"`);

    // 1. 执行 PCA 可视化
    console.log('\n--- 正在生成向量并执行 PCA 可视化 ---');
    const extractor = await getExtractor();
    const queryEmbedding = (await getEmbeddings(query, extractor))[0];
    const docEmbeddings = await getEmbeddings(documents, extractor);

    const allEmbeddings = [queryEmbedding!, ...docEmbeddings];
    const allDocs = [`[查询] ${query}`, ...documents];
    const categories = ['Query', ...documents.map(() => 'Document')];
    
    performPCA(allEmbeddings, allDocs, categories);

    // 2. 执行检索
    console.log('\n--- 余弦相似度 (Cosine Similarity) ---');
    const cosineResults = await retrieveRelevant(query, documents, 'cosine_similarity');
    cosineResults.forEach(r => console.log(`${r.score.toFixed(4)}: ${r.document}`));
    // 注意：
    // 尽管我们查询的是“亚洲”，但最佳匹配可能是“班夫国家公园”（加拿大）。
    // 原因：
    // 1. 模型局限性：paraphrase-multilingual-MiniLM-L12-v2 是一个小型的多语言模型，
    //    它可能更关注“旅游”、“推荐”、“景色”等语义匹配，而对“亚洲”这个地理实体的
    //    排他性理解不足。它可能认为“班夫”的描述与“值得一去”的语义非常接近。
    // 2. 缺乏显式过滤：纯向量检索是基于语义相似度的。如果文档中没有明确提到“亚洲”
    //    这个词（或者模型不知道富士山/长城属于亚洲），它就很难区分地理位置。
    // 
    // 解决方案：
    // 1. 使用更强大的模型（如 BGE-M3），它对实体概念理解更好。
    // 2. 混合检索（Hybrid Search）：结合关键词检索（BM25）和向量检索。
    // 3. 元数据过滤（Metadata Filtering）：给文档打上 { region: "Asia" } 标签，检索前先过滤。
    console.log(`\n最佳匹配 (余弦相似度): ${cosineResults[0]?.score.toFixed(4)}: ${cosineResults[0]?.document}`);

    console.log('\n--- 欧几里得距离 (Euclidean Distance) ---');
    const euclideanResults = await retrieveRelevant(query, documents, 'euclidean');
    euclideanResults.forEach(r => console.log(`${r.score.toFixed(4)}: ${r.document}`));
    if (euclideanResults.length > 0) {
        console.log(`\n最佳匹配 (欧几里得距离): ${euclideanResults[0]?.score.toFixed(4)}: ${euclideanResults[0]?.document}`);
    }
})();
