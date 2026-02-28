/// <reference lib="dom" />
/// <reference lib="es2020" />

declare const Chart: any;

interface TokenLogit {
    token: string;
    logit: number;
    prob: number; // Final probability after all transforms
}

// 1. 初始模拟数据 (Logits)
// 代表模型对下一个 Token 预测的原始输出分数
const initialTokens: TokenLogit[] = [
    { token: "生活", logit: 12.0, prob: 0 },   // 最常见搭配："改变我们的生活"
    { token: "未来", logit: 11.2, prob: 0 },   // 紧随其后："改变我们的未来"
    { token: "世界", logit: 10.4, prob: 0 },   // 依然很强："改变我们的世界"
    { token: "工作", logit: 9.6, prob: 0 },    // 常见领域："改变我们的工作"
    { token: "思维", logit: 6.0, prob: 0 },    // 抽象概念："改变我们的思维"
    { token: "午餐", logit: 2.0, prob: 0 },    // 离谱："改变我们的午餐" (除非是外卖AI)
];

// 2. 配置状态
let state = {
    temperature: 1.0,
    topK: 0, // 0 表示禁用
    topP: 1.0 // 1.0 表示禁用
};

// 3. 算法实现

function softmax(logits: number[], temperature: number): number[] {
    // 当温度极低时，直接返回 Argmax 结果 (One-Hot 分布)
    if (temperature < 0.05) {
        const maxLogit = Math.max(...logits);
        // 如果有多个相同的最大值，这里简单地只取第一个，或者平分概率
        // 为了简单，我们只取第一个匹配的
        const maxIndex = logits.indexOf(maxLogit);
        return logits.map((_, i) => (i === maxIndex ? 1 : 0));
    }

    // 应用温度系数
    const scaledLogits = logits.map(l => l / temperature);
    
    // 数值稳定性处理：减去最大值
    const maxLogit = Math.max(...scaledLogits);
    const exps = scaledLogits.map(l => Math.exp(l - maxLogit));
    
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
}

function processProbabilities() {
    // A. 应用温度系数 (Temperature) 和 Softmax
    let currentProbs = softmax(initialTokens.map(t => t.logit), state.temperature);
    
    // 创建临时数组进行处理（保留原始索引）
    let workingTokens = initialTokens.map((t, i) => ({
        ...t,
        prob: currentProbs[i] ?? 0,
        originalIndex: i
    }));

    // 按概率降序排序，以便进行 Top-K/Top-P 操作
    workingTokens.sort((a, b) => b.prob - a.prob);

    // B. 应用 Top-K
    if (state.topK > 0 && state.topK < workingTokens.length) {
        // 保留前 K 个，其余置为 0
        for (let i = 0; i < workingTokens.length; i++) {
            if (i >= state.topK) {
                const token = workingTokens[i];
                if (token) token.prob = 0;
            }
        }
        // 重新归一化
        const sum = workingTokens.reduce((acc, t) => acc + t.prob, 0);
        if (sum > 0) workingTokens.forEach(t => t.prob /= sum);
    }

    // C. 应用 Top-P (Nucleus Sampling)
    // 注意：如果有 Top-K，通常先执行 Top-K 再执行 Top-P
    if (state.topP < 1.0) {
        let cumulativeProb = 0;
        let cutOffIndex = -1;

        for (let i = 0; i < workingTokens.length; i++) {
            const token = workingTokens[i];
            if (!token) continue;
            
            cumulativeProb += token.prob;
            if (cumulativeProb >= state.topP) {
                cutOffIndex = i;
                break;
            }
        }

        // 如果找到了截断点，将之后的所有概率置为 0
        if (cutOffIndex !== -1 && cutOffIndex < workingTokens.length - 1) {
            for (let i = cutOffIndex + 1; i < workingTokens.length; i++) {
                const token = workingTokens[i];
                if (token) token.prob = 0;
            }
            // 重新归一化
            const sum = workingTokens.reduce((acc, t) => acc + t.prob, 0);
            if (sum > 0) workingTokens.forEach(t => t.prob /= sum);
        }
    }

    // 返回排序后的结果以便图表展示
    return workingTokens; 
}

// 4. 图表逻辑
let chartInstance: any = null;

function initChart() {
    const ctx = document.getElementById('probChart') as HTMLCanvasElement;
    if (!ctx) return;

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    type: 'line',
                    label: '分布曲线',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    tension: 0.4, // 平滑曲线
                    pointRadius: 0, // 隐藏数据点，使曲线更干净
                    fill: false
                },
                {
                    type: 'bar',
                    label: 'Token 概率',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: '概率 (Probability)'
                    }
                }
            },
            animation: {
                duration: 300 // 快速动画
            }
        }
    });
}

function updateChart() {
    const processed = processProbabilities();
    
    // 更新图表数据
    chartInstance.data.labels = processed.map(t => t.token);
    chartInstance.data.datasets[0].data = processed.map(t => t.prob); // 曲线
    chartInstance.data.datasets[1].data = processed.map(t => t.prob); // 柱状图
    
    // 可选：可以用不同颜色高亮显示被过滤掉的 Token (prob = 0)
    // 目前保持简单的蓝色柱状图
    
    chartInstance.update();
}

// Render Raw Logits
function renderLogits() {
    const container = document.getElementById('logits-table-container');
    if (!container) return;

    container.innerHTML = '';
    
    // Sort by logit descending just in case, though initialTokens is already sorted
    const sortedTokens = [...initialTokens].sort((a, b) => b.logit - a.logit);
    
    // Calculate raw probabilities (Temp=1.0)
    const rawProbs = softmax(sortedTokens.map(t => t.logit), 1.0);

    sortedTokens.forEach((t, i) => {
        const div = document.createElement('div');
        div.style.cssText = `
            background: white;
            border: 1px solid #ffcc80;
            border-radius: 6px;
            padding: 8px 12px;
            min-width: 90px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        `;
        
        const prob = rawProbs[i] ?? 0;
        const probPercent = (prob * 100).toFixed(1) + '%';
        
        div.innerHTML = `
            <div style="font-weight: bold; color: #333; margin-bottom: 4px;">${t.token}</div>
            <div style="font-family: monospace; color: #e65100; font-size: 0.9em; margin-bottom: 2px;">Logit: ${t.logit.toFixed(1)}</div>
            <div style="font-family: monospace; color: #1976d2; font-size: 0.85em;">Prob: ${probPercent}</div>
        `;
        
        container.appendChild(div);
    });
}

// 5. UI 事件监听器
document.addEventListener('DOMContentLoaded', () => {
    // 获取元素
    const tempInput = document.getElementById('temperature') as HTMLInputElement;
    const topKInput = document.getElementById('top-k') as HTMLInputElement;
    const topPInput = document.getElementById('top-p') as HTMLInputElement;
    const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;

    const tempVal = document.getElementById('temp-val')!;
    const topKVal = document.getElementById('top-k-val')!;
    const topPVal = document.getElementById('top-p-val')!;

    // 初始化
    initChart();
    renderLogits(); // Show raw logits
    updateChart();

    // 事件处理
    function updateState() {
        state.temperature = parseFloat(tempInput.value);
        state.topK = parseInt(topKInput.value);
        state.topP = parseFloat(topPInput.value);

        // 更新显示的数值
        tempVal.textContent = state.temperature.toFixed(1);
        topKVal.textContent = state.topK === 0 ? '0 (关闭)' : state.topK.toString();
        topPVal.textContent = state.topP === 1.0 ? '1.0 (关闭)' : state.topP.toFixed(2);

        updateChart();
    }

    tempInput.addEventListener('input', updateState);
    topKInput.addEventListener('input', updateState);
    topPInput.addEventListener('input', updateState);

    resetBtn.addEventListener('click', () => {
        tempInput.value = "1.0";
        topKInput.value = "0";
        topPInput.value = "1.0";
        updateState();
    });
});
