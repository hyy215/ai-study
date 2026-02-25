
// --- 基础数学工具类 (模拟 PyTorch 的 Tensor 运算) ---

export type Matrix = number[][];
export type Vector = number[];

export class TensorMath {
    /**
     * 创建一个指定形状的随机矩阵 (模拟 torch.rand)
     */
    static rand(rows: number, cols: number): Matrix {
        return Array.from({ length: rows }, () => 
            Array.from({ length: cols }, () => Math.random())
        );
    }

    /**
     * 矩阵转置 (模拟 tensor.transpose)
     */
    static transpose(matrix: Matrix): Matrix {
        if (!matrix.length || !matrix[0]) {
            return [];
        }
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]!));
    }

    /**
     * 矩阵乘法 (模拟 torch.matmul)
     * A (m x n) * B (n x p) -> C (m x p)
     */
    static matmul(a: Matrix, b: Matrix): Matrix {
        if (!a.length || !a[0] || !b.length || !b[0]) {
            throw new Error("Invalid matrix dimensions: Matrices cannot be empty");
        }

        const m = a.length;
        const n = a[0].length;
        const p = b[0].length;
        
        if (n !== b.length) {
            throw new Error(`Dimension mismatch: A columns (${n}) != B rows (${b.length})`);
        }

        const result: Matrix = Array.from({ length: m }, () => Array(p).fill(0));

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    // 使用非空断言 ! 因为我们在前面已经检查了维度
                    sum += a[i]![k]! * b[k]![j]!;
                }
                result[i]![j] = sum;
            }
        }
        return result;
    }

    /**
     * Softmax 函数 (模拟 F.softmax)
     * 对矩阵的每一行进行 Softmax 归一化
     */
    static softmax(matrix: Matrix): Matrix {
        return matrix.map(row => {
            const maxVal = Math.max(...row); // 防止溢出，减去最大值
            const exps = row.map(x => Math.exp(x - maxVal));
            const sumExps = exps.reduce((a, b) => a + b, 0);
            return exps.map(x => x / sumExps);
        });
    }

    /**
     * 标量除法 (模拟 tensor / sqrt(d_k))
     */
    static scale(matrix: Matrix, scalar: number): Matrix {
        return matrix.map(row => row.map(val => val / scalar));
    }

    /**
     * 生成下三角矩阵掩码 (模拟 torch.tril)
     * 1 表示保留，0 表示被 mask
     */
    static tril(size: number): Matrix {
        return Array.from({ length: size }, (_, i) => 
            Array.from({ length: size }, (_, j) => (j <= i ? 1 : 0))
        );
    }

    /**
     * 掩码填充 (模拟 tensor.masked_fill)
     * 如果 mask 对应位置为 0，则将原矩阵对应位置的值替换为 value (通常是 -infinity)
     */
    static maskedFill(matrix: Matrix, mask: Matrix, value: number): Matrix {
        return matrix.map((row, i) => 
            row.map((val, j) => (mask[i]![j] === 0 ? value : val))
        );
    }

    /**
     * 矩阵拼接 (模拟 torch.cat dim=-1)
     * 将多个矩阵在最后一个维度(列)上拼接
     * 假设所有矩阵的行数相同
     */
    static concat(matrices: Matrix[]): Matrix {
        if (matrices.length === 0) return [];
        const rows = matrices[0]!.length;
        
        return Array.from({ length: rows }, (_, i) => {
            return matrices.reduce((acc, curr) => acc.concat(curr[i]!), [] as number[]);
        });
    }
}
