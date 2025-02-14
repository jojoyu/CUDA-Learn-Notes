#include <iostream>

void MatrixMulOnHost(int m, int n, int k, float* A, float* B, float* C)
{
    for (int Row = 0; Row < m; ++Row)
    {
        for (int Col = 0; Col < k; ++Col)
        {
            float sum = 0;
            for (int i = 0; i < n; ++i)
            {
                float a = A[Row * n + i];
                float b = B[i * k + Col];
                sum += a * b;
            }
            C[Row * k + Col] = sum;
        }
    }
}

int main()
{
    // 更大的矩阵维度
    int m = 3, n = 4, k = 2;

    // 初始化矩阵 A 和 B
    float A[m * n] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    float B[n * k] = {
        13, 14,
        15, 16,
        17, 18,
        19, 20
    };

    // 结果矩阵 C
    float C[m * k];

    // 调用矩阵乘法函数
    MatrixMulOnHost(m, n, k, A, B, C);

    // 打印结果矩阵 C
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            std::cout << C[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
