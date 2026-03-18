import weaviate, { EmbeddedOptions } from 'weaviate-ts-embedded';
import type {EmbeddedClient} from 'weaviate-ts-embedded';
import { generateUuid5 } from 'weaviate-ts-client'; 
import { mockData } from './mock-data.js';

// EmbeddedOptions.port 控制 Weaviate 二进制监听端口，但客户端连接地址需通过第二个参数显式指定
// 端口 9999：避免 WHATWG 禁止端口（6665-6669 为 IRC 端口，Node.js fetch 会拒绝）
const WEAVIATE_PORT = 9999;
const client: EmbeddedClient = weaviate.client(
  new EmbeddedOptions({
    port: WEAVIATE_PORT,
    version: '1.26.1', // Add version to prevent github API rate limit/connection issues
    env: {
      "ENABLE_MODULES": "text2vec-transformers, reranker-transformers",
      "ENABLE_API_BASED_MODULES": "true",
      "TRANSFORMERS_INFERENCE_API": "http://127.0.0.1:8080",
      "RERANKER_INFERENCE_API": "http://127.0.0.1:8080",
      "LOG_LEVEL": "error", // 只显示错误日志，隐藏 info 级别的 JSON 日志
      "DISABLE_TELEMETRY": "true", // 禁用遥测
      "LOG_FORMAT": "text" // 使用文本格式输出日志，避免 JSON 格式
    }
  }),
  { host: `127.0.0.1:${WEAVIATE_PORT}`, scheme: 'http' }  // 显式指定客户端连接地址
);

// 辅助函数：打印查询结果
function printResults(header: string, results: any[], formatter: (res: any) => string) {
  console.log(`\n--- ${header} ---`);
  if (results && results.length > 0) {
    results.forEach((res, i) => {
      console.log(`${i+1}. ${formatter(res)}`);
      console.log(`   描述: ${res.description}`);
    });
  } else {
    console.log("未找到符合条件的结果。");
  }
}

async function main() {
  console.log("正在启动嵌入式 Weaviate (v3)...");
  await client.embedded.start();

  const className = "TravelDestination";

  try {
    // --- 1. Schema 管理 ---
    const exists = await client.schema.exists(className);
    if (exists) {
      console.log(`清理旧集合: ${className}`);
      await client.schema.classDeleter().withClassName(className).do();
    }

    console.log(`正在创建集合: ${className}`);
    await client.schema.classCreator().withClass({
      class: className,
      vectorizer: "text2vec-transformers",
      moduleConfig: {
        "text2vec-transformers": {
          vectorizeClassName: false,
          poolingStrategy: "masked_mean"
        },
        "reranker-transformers": {}
      },
      // 旧版 GraphQL API 中，rerank 能否出现在 _additional 里取决于类是否配置了 reranker 模块
      properties: [
        { name: "place", dataType: ["text"] },
        { name: "state", dataType: ["text"] },
        { name: "description", dataType: ["text"] },
        { name: "best_season_to_visit", dataType: ["text"] },
        { name: "attractions", dataType: ["text"] },
        {
          name: "budget",
          dataType: ["text"],
          moduleConfig: { "text2vec-transformers": { skip: true } }
        },
        {
          name: "user_ratings",
          dataType: ["number"],
          moduleConfig: { "text2vec-transformers": { skip: true } }
        },
        {
          name: "last_updated",
          dataType: ["text"],
          moduleConfig: { "text2vec-transformers": { skip: true } }
        }
      ]
    }).do();

    // --- 2. 批量导入数据 ---
    console.log("开始批量导入数据...");
    let batcher = client.batch.objectsBatcher();
    
    for (const item of mockData) {
      batcher = batcher.withObject({
        class: className,
        id: generateUuid5(item.place), // 确定性 UUID
        properties: { ...item }
      });
    }

    const batchRes = await batcher.do();
    
    // 检查导入是否有错
    const errors = batchRes.filter(r => r.result?.errors);
    if (errors.length > 0) {
      console.error("导入出错详情:", JSON.stringify((errors as any)[0].result?.errors, null, 2));
    } else {
      console.log(`✅ 成功导入 ${batchRes.length} 条记录`);
    }

    // --- 3. 标量筛选测试 (GraphQL) ---
    // 等同于 SQL: SELECT place, description, user_ratings FROM TravelDestination WHERE user_ratings >= 4.8 LIMIT 2;
    const searchResult = await client.graphql
      .get()
      .withClassName(className)
      .withFields("place description user_ratings")
      .withWhere({
        path: ["user_ratings"],
        operator: "GreaterThanEqual",
        valueNumber: 4.8
      })
      .withLimit(2)
      .do();

    const results = searchResult.data.Get[className];
    printResults("标量筛选测试: user_ratings >= 4.8", results, (res: any) => `${res.place} (评分: ${res.user_ratings})`);

    // --- 4. BM25 关键字搜索 (结合标量过滤) ---
    const bm25Query = "秋季";
    const bm25Result = await client.graphql
      .get()
      .withClassName(className)
      .withFields("place description best_season_to_visit budget _additional { score }")
      .withBm25({
        query: bm25Query,
      })
      .withWhere({
        path: ["budget"],
        operator: "ContainsAny",
        valueTextArray: ["低", "中等"]
      })
      .withLimit(4)
      .do();

    const bm25Results = bm25Result.data.Get[className];
    printResults(
      `BM25 关键字搜索: "${bm25Query}" + 预算过滤`, 
      bm25Results, 
      (res: any) => `${res.place} (预算: ${res.budget}, 季节: ${res.best_season_to_visit}, BM25分数: ${res._additional.score})`
    );

    // --- 5. 混合搜索 (Hybrid Search: 向量搜索 + 关键字搜索) ---
    const hybridQuery = "想去便宜的地方看雪";
    const hybridResult = await client.graphql
      .get()
      .withClassName(className)
      .withFields("place description best_season_to_visit budget _additional { score }")
      .withHybrid({
        query: hybridQuery, // 融合了意图(语义)和关键字
        alpha: 0.3 // 0=纯关键字搜索(BM25)，1=纯向量搜索，0.3 偏向关键字
      })
      .withWhere({
        path: ["budget"],
        operator: "ContainsAny",
        valueTextArray: ["低", "中等"]
      })
      .withLimit(4)
      .do();

    const hybridResults = hybridResult.data.Get[className];
    printResults(
      `混合搜索 (Hybrid): "${hybridQuery}" (alpha: 0.3)`,
      hybridResults, 
      (res: any) => `${res.place} (预算: ${res.budget}, 季节: ${res.best_season_to_visit}, 混合分数: ${res._additional.score})`
    );

    // --- 6. 语义搜索 + Rerank ---
    // 等价于 Python:
    // collection.query.near_text(query=..., limit=5, rerank=Rerank(prop="attractions", query="Fun places"))
    const nearTextQuery = "我想找适合冬天旅行、便宜又好玩的地方";
    const rerankQuery = "好玩的景点";
    const rerankResult = await client.graphql
      .get()
      .withClassName(className)
      .withFields(`
        place
        state
        description
        attractions
        budget
        best_season_to_visit
        _additional {
          distance
          rerank(
            property: "attractions"
            query: "${rerankQuery}"
          ) {
            score
          }
        }
      `)
      .withNearText({
        concepts: [nearTextQuery]
      })
      .withLimit(5)
      .do();

    const rerankResults = rerankResult.data.Get[className];
    printResults(
      `NearText + Rerank: 语义搜索 "${nearTextQuery}" -> 重排 "${rerankQuery}"`,
      rerankResults,
      (res: any) => `${res.place} (景点: ${res.attractions}, 重排分数: ${res._additional?.rerank?.[0]?.score ?? "N/A"}, 距离: ${res._additional?.distance ?? "N/A"})`
    );

  } catch (error) {
    console.error("❌ 运行过程中发生错误:", error);
  } finally {
    // 开发环境可以注释掉 stop，方便手动查看数据
    await client.embedded.stop();
  }
}

main();