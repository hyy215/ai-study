
import { getEmbeddings, getExtractor } from '../vector-embeddings/retrieval.js';
import { cosineSimilarity } from '../vector-embeddings/vectorUtils.js';
import { precisionAtK, recallAtK } from './metrics.js';

// 1. 模拟数据集 (替代 sklearn fetch_20newsgroups)
// 包含: text, category
interface Document {
    text: string;
    category: string;
}

const dataset: Document[] = [
    // sci.space (太空科学)
    { text: "NASA 计划明年向火星发射新的探测器。", category: "sci.space" },
    { text: "哈勃望远镜捕捉到了遥远星系的惊人图像。", category: "sci.space" },
    { text: "SpaceX 成功回收了猎鹰 9 号火箭。", category: "sci.space" },
    { text: "国际空间站的宇航员正在进行零重力实验。", category: "sci.space" },
    { text: "对黑洞的研究揭示了宇宙的奥秘。", category: "sci.space" },
    
    // comp.graphics (计算机图形学)
    { text: "光线追踪是一种生成逼真图像的技术。", category: "comp.graphics" },
    { text: "OpenGL 和 DirectX 是流行的图形 API。", category: "comp.graphics" },
    { text: "Blender 等 3D 建模软件用于动画制作。", category: "comp.graphics" },
    { text: "GPU 加速提高了渲染性能。", category: "comp.graphics" },
    { text: "纹理映射为 3D 表面增加了细节。", category: "comp.graphics" },

    // rec.sport.hockey (冰球运动)
    { text: "守门员在最后一分钟做出了不可思议的扑救。", category: "rec.sport.hockey" },
    { text: "NHL 季后赛将于下周开始。", category: "rec.sport.hockey" },
    { text: "冰球需要很强的滑冰技巧。", category: "rec.sport.hockey" },
    { text: "球队打进了一个多打少进球。", category: "rec.sport.hockey" },
    { text: "冰球杆由复合材料制成。", category: "rec.sport.hockey" },

    // sci.med (医学科学)
    { text: "心血管锻炼对心脏健康有益。", category: "sci.med" },
    { text: "针对该病毒的新疫苗正在开发中。", category: "sci.med" },
    { text: "医生建议均衡饮食和规律睡眠。", category: "sci.med" },
    { text: "MRI 扫描有助于诊断内部损伤。", category: "sci.med" },
    { text: "抗生素用于治疗细菌感染。", category: "sci.med" }
];

// 测试查询
const testQueries = [
    { query: "太空探索技术的进步", desired_category: "sci.space" },
    { query: "计算机图形学中的实时渲染技术", desired_category: "comp.graphics" },
    { query: "心血管医学研究的最新发现", desired_category: "sci.med" },
    { query: "NHL 季后赛和球队表现统计", desired_category: "rec.sport.hockey" }
];

/**
 * 获取前 K 个最大值的索引
 */
function topKGreatestIndices(list: number[], k: number): number[] {
    return list
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value)
        .slice(0, k)
        .map(item => item.index);
}

/**
 * 计算评估指标
 */
async function computeMetrics(queries: typeof testQueries, docEmbeddings: number[][], topK: number = 5) {
    const results = [];
    const extractor = await getExtractor();

    for (const item of queries) {
        const { query, desired_category } = item;

        // 生成查询向量
        const queryEmbedding = (await getEmbeddings(query, extractor))[0];

        // 计算相似度 (Cosine Similarity)
        // 注意: cosineSimilarity 返回 number[]
        const scores = cosineSimilarity(queryEmbedding!, docEmbeddings);

        // 获取前 K 个结果的索引
        const topIndices = topKGreatestIndices(scores, topK);

        // 获取检索到的类别
        const retrievedCategories = topIndices.map(idx => dataset[idx]!.category);

        // 计算指标
        const relevantInTopK = retrievedCategories.filter(cat => cat === desired_category).length;
        const totalRelevantInCorpus = dataset.filter(doc => doc.category === desired_category).length;

        const p = precisionAtK(relevantInTopK, topK);
        const r = recallAtK(relevantInTopK, totalRelevantInCorpus);

        results.push({
            query,
            precision: p,
            recall: r,
            retrieved: retrievedCategories // 用于调试
        });
    }

    return results;
}

/**
 * 主执行函数
 */
async function runEvaluation() {
    console.log(`数据集大小: ${dataset.length}`);
    console.log(`测试查询数量: ${testQueries.length}`);

    // 1. 预计算所有文档的向量 (模拟 embeddings.joblib)
    console.log("\n正在生成文档向量库...");
    const extractor = await getExtractor();
    const docTexts = dataset.map(d => d.text);
    // 批量生成向量
    const docEmbeddings = await getEmbeddings(docTexts, extractor);
    console.log("文档向量库生成完毕。");

    // 2. 对不同的 K 值进行评估
    const kValues = [1, 3, 5];

    for (const k of kValues) {
        console.log(`\n${'='.repeat(80)}`);
        console.log(`评估结果 (K=${k}):`);
        console.log('='.repeat(80));

        const results = await computeMetrics(testQueries, docEmbeddings, k);

        let totalPrecision = 0;
        let totalRecall = 0;

        for (const res of results) {
            console.log(`查询: "${res.query}"`);
            console.log(`  Precision@${k}: ${res.precision.toFixed(2)}, Recall@${k}: ${res.recall.toFixed(2)}`);
            // console.log(`  Retrieved: ${res.retrieved.join(', ')}`); // 可选：打印检索到的类别
            console.log();
            
            totalPrecision += res.precision;
            totalRecall += res.recall;
        }

        // 打印平均指标
        const avgPrecision = totalPrecision / results.length;
        const avgRecall = totalRecall / results.length;
        console.log(`>>> 平均 Precision@${k}: ${avgPrecision.toFixed(4)}`);
        console.log(`>>> 平均 Recall@${k}:    ${avgRecall.toFixed(4)}`);
    }
}

// 运行
runEvaluation();
