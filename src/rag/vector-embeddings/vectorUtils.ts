
/**
 * RAG (检索增强生成) 的向量工具函数
 * 实现了向量搜索中常用的标准距离度量算法。
 */

/**
 * 计算两个向量的点积。
 * 公式: a · b = ∑(ai * bi)
 */
export function dotProduct(v1: number[], v2: number[]): number {
    if (v1.length !== v2.length) {
        throw new Error(`向量长度不匹配: ${v1.length} vs ${v2.length}`);
    }
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        sum += v1[i]! * v2[i]!;
    }
    return sum;
}

/**
 * 计算向量的 L2 范数 (欧几里得范数)。
 * 公式: ||v|| = √(∑vi²)
 */
export function norm(v: number[]): number {
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
        sum += v[i]! * v[i]!;
    }
    return Math.sqrt(sum);
}

/**
 * 计算一个向量与一组向量之间的余弦相似度。
 * 返回相似度列表。
 * 公式: cos(θ) = (A · B) / (||A|| * ||B||)
 */
export function cosineSimilarity(v1: number[], arrayOfVectors: number[] | number[][]): number[] {
    // 规范化输入，使其始终为向量数组
    const vectors = Array.isArray(arrayOfVectors[0]) 
        ? (arrayOfVectors as number[][]) 
        : [arrayOfVectors as number[]];

    const v1Norm = norm(v1);

    return vectors.map(v2 => {
        const dot = dotProduct(v1, v2);
        const v2Norm = norm(v2);
        if (v1Norm === 0 || v2Norm === 0) return 0; // 处理零向量的情况
        return dot / (v1Norm * v2Norm);
    });
}

/**
 * 计算一个向量与一组向量之间的欧几里得距离。
 * 返回距离列表。
 * 公式: d(x, y) = √(∑(xi - yi)²)
 */
export function euclideanDistance(v1: number[], arrayOfVectors: number[] | number[][]): number[] {
    // 规范化输入，使其始终为向量数组
    const vectors = Array.isArray(arrayOfVectors[0]) 
        ? (arrayOfVectors as number[][]) 
        : [arrayOfVectors as number[]];

    return vectors.map(v2 => {
        if (v1.length !== v2.length) {
            throw new Error(`形状不匹配: v1 长度 ${v1.length}, v2 长度 ${v2.length}`);
        }
        
        let sumSq = 0;
        for (let i = 0; i < v1.length; i++) {
            sumSq += Math.pow(v1[i]! - v2[i]!, 2);
        }
        return Math.sqrt(sumSq);
    });
}
