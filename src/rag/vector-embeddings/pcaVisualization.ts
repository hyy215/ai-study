
import { PCA } from 'ml-pca';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

/**
 * 执行 PCA 将嵌入向量降维到 2 维，并生成可视化数据文件。
 * @param embeddings 高维嵌入向量 (N x D)
 * @param documents 对应的文档文本标签 (N)
 * @param categories 每个文档的可选类别 (N)
 */
export function performPCA(embeddings: number[][], documents: string[], categories?: string[]) {
    try {
        // 执行 PCA 将维度降至 2
        const pca = new PCA(embeddings);
        const newPoints = pca.predict(embeddings, { nComponents: 2 }).to2DArray();

        // 准备 JSON 数据
        const data = documents.map((doc, i) => {
            const point = newPoints[i];
            if (!point || point.length < 2) return null;
            const [x, y] = point;

            let category = 'Unknown';
            if (categories && categories[i]) {
                category = categories[i]!;
            }

            return {
                label: doc,
                x: x,
                y: y,
                category: category
            };
        }).filter(item => item !== null);

        // 生成 JS 文件内容
        const jsContent = `// 自动生成的可视化数据 (由 pcaVisualization.ts 生成)
const vizData = ${JSON.stringify(data, null, 2)};
`;

        // 写入文件
        const __dirname = path.dirname(fileURLToPath(import.meta.url));
        const outputPath = path.join(__dirname, 'viz-data.js');
        
        fs.writeFileSync(outputPath, jsContent, 'utf-8');
        console.log(`\n可视化数据已写入: ${outputPath}`);
        console.log('请打开 viz.html 查看图表。');

        // 同时打印简单的 CSV 到控制台（保留原有功能，方便快速查看）
        console.log('\n--- PCA CSV 预览 ---');
        console.log('Label,X,Y,Category');
        data.slice(0, 5).forEach(item => { // 只打印前5行避免刷屏
            if(!item) return;
            const cleanLabel = item.label.replace(/,/g, '，').substring(0, 20);
            console.log(`"${cleanLabel}",${item.x!.toFixed(4)},${item.y!.toFixed(4)},${item.category}`);
        });
        console.log('...');

        console.log('\n--- Explained Variance (解释方差) ---');
        console.log(pca.getExplainedVariance());

    } catch (error) {
        console.error('运行 PCA 时出错:', error);
    }
}
