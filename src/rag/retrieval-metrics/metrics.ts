
/**
 * 计算检索系统的 Precision@K (前 K 个结果的准确率)。
 *
 * Precision@K = (前 K 个结果中相关文档的数量) / K
 *
 * @param relevantCount 前 K 个结果中相关文档的数量
 * @param k 检索回来的文档总数 (K)
 */
export function precisionAtK(relevantCount: number, k: number): number {
    if (relevantCount < 0 || k < 0) {
        throw new Error("所有输入值必须非负。");
    }

    if (k === 0) {
        return 0.0;
    }

    return relevantCount / k;
}

/**
 * 计算检索系统的 Recall@K (前 K 个结果的召回率)。
 *
 * Recall@K = (前 K 个结果中相关文档的数量) / (语料库中相关文档的总数)
 *
 * @param relevantCount 前 K 个结果中相关文档的数量
 * @param totalRelevant 语料库中相关文档的总数
 */
export function recallAtK(relevantCount: number, totalRelevant: number): number {
    if (relevantCount < 0 || totalRelevant < 0) {
        throw new Error("所有输入值必须非负。");
    }

    if (totalRelevant === 0) {
        return 0.0;
    }

    return relevantCount / totalRelevant;
}
