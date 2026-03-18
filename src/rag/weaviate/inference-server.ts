import express from 'express';
import { pipeline, FeatureExtractionPipeline, TextClassificationPipeline } from '@huggingface/transformers';

const app = express();

app.use((req, _, next) => {
  if (!req.headers['content-type']) {
    req.headers['content-type'] = 'application/json';
  }
  next();
});

// Weaviate text2vec-transformers 模块有时会在发送 application/json 时不带正确的字符集
// 或者根本不带 Content-Type。
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

let extractor: FeatureExtractionPipeline | null = null;
let reranker: TextClassificationPipeline | null = null;

// 加载模型
async function initModels() {
  console.log("正在加载本地模型，请稍候...");
  extractor = (await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2')) as any as FeatureExtractionPipeline;
  reranker = (await pipeline('text-classification', 'Xenova/bge-reranker-base')) as any as TextClassificationPipeline;
  console.log("✅ 所有模型已就绪。");
}

// 向量化接口：Weaviate text2vec-transformers 默认会请求 /vectors
app.post('/vectors', async (req, res) => {
  try {
    const text = req.body?.text || req.body?.texts;
    if (!text || !extractor) {
      res.status(400).json({ error: "Missing text or model", body: req.body });
      return;
    }

    const texts = Array.isArray(text) ? text : [text];
    const output = await extractor(texts, { pooling: 'mean', normalize: true });
    
    // Weaviate 期望返回: { "text": "...", "vector": [0.1, ...] }
    // 如果是批量，通常返回第一条或按协议平铺
    res.json({
      text: texts[0],
      vector: Array.from(output.data) 
    });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 重排接口：Weaviate reranker-transformers 插件协议
app.post('/rerank', async (req, res) => {
  try {
    const { query, documents } = req.body;
    if (!reranker) {
      res.status(503).send("Reranker not ready");
      return;
    }

    // Weaviate 发送的 documents 是纯字符串数组（来自 ranker.go RankInput.Documents []string）
    const docTexts = (documents as (string | { text: string })[]).map(doc =>
      typeof doc === 'string' ? doc : doc.text
    );

    // cross-encoder pipeline 需要 {text: query, text_pair: document} 格式
    const inputs = docTexts.map((docText: string) => ({ text: query, text_pair: docText }));
    const rawResults: any = await (reranker as any)(inputs);

    // Weaviate ranker.go 期望响应格式为：
    // { "query": "...", "scores": [{ "document": "...", "score": 0.9 }, ...] }
    // 对应 Go 结构体 RankResponse { Query, Scores []DocumentScore }
    const scores = docTexts.map((docText: string, i: number) => {
      const r = Array.isArray(rawResults[i]) ? rawResults[i][0] : rawResults[i];
      return { document: docText, score: r.score as number };
    });

    res.json({ query, scores });
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 健康检查
app.get('/.well-known/ready', (_req, res) => {
  res.sendStatus(200);
});

app.listen(8080, '127.0.0.1', async () => {
  await initModels();
  console.log("🚀 推理服务器运行在 http://127.0.0.1:8080");
});