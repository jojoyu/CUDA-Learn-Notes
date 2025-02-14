#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#include <limits>
#include <type_traits>
#include <chrono>
#include <iomanip>


#define TILE_WIDTH 2

// https://zhuanlan.zhihu.com/p/342103911 参考原理

template<typename T>
bool isEqual(T a, T b) {
    if constexpr (std::is_floating_point<T>::value) {
        const T epsilon = std::numeric_limits<T>::epsilon();
        return std::fabs(a - b) <= epsilon * std::max(std::fabs(a), std::fabs(b));
    } else {
        return a == b;
    }
}

template<typename T>
bool isEqualRelative(T a, T b, T epsilon = 1e-6 /* std::numeric_limits<T>::epsilon() */) {
    if constexpr (std::is_floating_point<T>::value) {
        T diff = std::fabs(a - b);
        T maxVal = std::max(std::fabs(a), std::fabs(b));
        return diff <= epsilon * maxVal;
    } else {
        return a == b;
    }
}

/* 参考这段代码理解MatrixMulKernel
__global__ void matrixMul(A_gpu, B_gpu, C_gpu, K){
   // A_gpu, B_gpu in global memory

    temp <= 0

    i <= blockIdx.y * blockDim.y + threadIdx.y    // Row i of matrix C
    j <= blockIdx.x * blockDim.x + threadIdx.x    // Column j of matrix C

    for k = 0 to K-1 do
        accu <= accu + A_gpu(i, k) * B_gpu(k, j)
    end

    C_gpu(i, j) <= accu

}
*/

// GPU内核函数定义，很普通没有考虑的内存访问优化的情况
__global__ void MatrixMulKernel(int m, int n, int k, float *A, float *B, float *C) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < m) && (Col < k)) {
        float Cvalue = 0.0;
        for (int i = 0; i < n; ++i) {
            Cvalue += A[Row * n + i] * B[i * k + Col]; // 修正了这里的索引
        }
        C[Row * k + Col] = Cvalue;
    }
}

// /* 参考这段代码理解 MatrixMulKernel_Tiling
// __global__ void matrixMul(A_gpu, B_gpu, C_gpu, K){

//     // A_tile, B_tile in shared memory
//     __shared__ float A_tile(blockDim.y, blockDim.x)
//     __shared__ float B_tile(blockDim.x, blockDim.y)

//     // accu in register
//     accu <= 0

//     /* Accumulate C tile by tile. */

//     for tileIdx = 0 to (K/blockDim.x - 1) do

//         /* Load one tile of A and one tile of B into shared mem */

//         // Row i of matrix A
//         i <= blockIdx.y * blockDim.y + threadIdx.y
//         // Column j of matrix A
//         j <= tileIdx * blockDim.x + threadIdx.x
//         // Load A(i, j) to shared mem
//         A_tile(threadIdx.y, threadIdx.x) <= A_gpu(i, j)
//         // Load B(j,i) to shared mem
//         B_tile(threadIdx.x, threadIdx.y) <= B_gpu(j, i) // Global Mem Not coalesced
//         // Synchronize before computation
//         __sync()

//         /* Accumulate one tile of C from tiles of A and B in shared mem */

//         for k = 0 to threadDim.x do
//             // Accumulate for matrix C
//             accu <= accu + A_tile(threadIdx.y, k) * B_tile(k, threadIdx.x)
//         end
//         // Synchronize
//         __sync()

//     end

//     // Row i of matrix C
//     i <= blockIdx.y * blockDim.y + threadIdx.y
//     // Column j of matrix C
//     j <= blockIdx.x * blockDim.x + threadIdx.x
//     // Store accumulated value to C(i, j)
//     C_gpu(i, j) <= accu

// }

#define DEBUG 1

// __global__ void MatrixMulKernel_Tiling(int m, int n, int k, float *A, float *B, float *C) {
//     __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

//     int bx = blockIdx.x,  by = blockIdx.y;
//     int tx = threadIdx.x, ty = threadIdx.y;
//     int Row = by * TILE_WIDTH + ty;
//     int Col = bx * TILE_WIDTH + tx;

//     float Cvalue = 0.0f;

//     #if DEBUG
//     printf("Block (%d, %d), Thread (%d, %d): Starting computation\n", bx, by, tx, ty);
//     #endif

//     for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t) {
//         if (Row < m && t * TILE_WIDTH + tx < k) {
//             A_tile[ty][tx] = A[Row * k + t * TILE_WIDTH + tx];
//             #if DEBUG
//             printf("Block (%d, %d), Thread (%d, %d): Loading A[%d] = %.2f to A_tile[%d][%d]\n", 
//                    bx, by, ty, tx, Row * k + t * TILE_WIDTH + tx, A_tile[ty][tx], ty, tx);
//             #endif
//         } else {
//             A_tile[ty][tx] = 0.0f;
//             #if DEBUG
//             printf("Block (%d, %d), Thread (%d, %d): Setting A_tile[%d][%d] = 0.0f\n", bx, by, ty, tx, ty, tx);
//             #endif
//         }

//         if (Col < n && t * TILE_WIDTH + ty < k) {
//             B_tile[ty][tx] = B[(t * TILE_WIDTH + ty) * n + Col];
//             #if DEBUG
//             printf("Block (%d, %d), Thread (%d, %d): Loading B[%d] = %.2f to B_tile[%d][%d]\n", 
//                    bx, by, ty, tx, (t * TILE_WIDTH + ty) * n + Col, B_tile[ty][tx], ty, tx);
//             #endif
//         } else {
//             B_tile[ty][tx] = 0.0f;
//             #if DEBUG
//             printf("Block (%d, %d), Thread (%d, %d): Setting B_tile[%d][%d] = 0.0f\n", bx, by, ty, tx, ty, tx);
//             #endif
//         }

//         __syncthreads();

//         for (int i = 0; i < TILE_WIDTH; ++i) {
//             Cvalue += A_tile[ty][i] * B_tile[i][tx];
//             #if DEBUG
//             printf("Block (%d, %d), Thread (%d, %d): Cvalue += A_tile[%d][%d] (%.2f) * B_tile[%d][%d] (%.2f) = %.2f\n", 
//                    bx, by, ty, tx, ty, i, A_tile[ty][i], i, tx, B_tile[i][tx], Cvalue);
//             #endif
//         }

//         __syncthreads();
//     }

//     if (Row < m && Col < n) {
//         C[Row * n + Col] = Cvalue;
//         #if DEBUG
//         printf("Block (%d, %d), Thread (%d, %d): Storing final Cvalue %.2f to C[%d]\n", 
//                bx, by, ty, tx, Cvalue, Row * n + Col);
//         #endif
//     }
// }

__global__ void MatrixMulKernel_Tiling_GPT(int m, int n, int k, float* A, float* B, float* C) {
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    // accu in register
    float accu = 0.0f;

    // Row i of matrix A
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    // Column j of matrix A
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    /* Accumulate C tile by tile. */
    for (int tileIdx = 0; tileIdx < (k / blockDim.x); tileIdx++) {
        /* Load one tile of A and one tile of B into shared mem */
        if ((i < m) && (threadIdx.x + tileIdx * TILE_WIDTH < n)) {
            // Load A(i, j) to shared mem
            A_tile[threadIdx.y][threadIdx.x] = A[i * n + threadIdx.x + tileIdx * blockDim.x];
            printf("1. Block (%d, %d), Thread (%d, %d): tileIdx (%d)"
                "Loaded A[%d](%.2f) to "
                "A_tile[%d][%d] = %.2f to shared mem\n", 
                bx, by, ty, tx, tileIdx,
                (i * n + threadIdx.x + tileIdx * blockDim.x), A[i * n + threadIdx.x + tileIdx * blockDim.x], 
                threadIdx.y, threadIdx.x, A_tile[threadIdx.y][threadIdx.x]);

        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
            printf("1. Block (%d, %d), Thread (%d, %d): ""Loaded A_tile 0, i:j(%d, %d).\n", bx, by, ty, tx, i, j);
        }

        if ((j < k) && (threadIdx.y + tileIdx * TILE_WIDTH < n)) {
            // Load B(i, j) to shared mem
            B_tile[threadIdx.y][threadIdx.x] = B[(threadIdx.y + tileIdx * blockDim.y ) * k + j];
            printf("2. Block (%d, %d), Thread (%d, %d): tileIdx (%d) "
                "Loaded B[%d](%.2f) "
                "to B_tile[%d][%d] = %.2f to shared mem\n", 
                bx, by, ty, tx, tileIdx,
                (threadIdx.y + tileIdx * blockDim.y) * k + j, B[(threadIdx.y + tileIdx * blockDim.y) * k + j], 
                threadIdx.y, threadIdx.x, B_tile[threadIdx.y][threadIdx.x]);
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
            printf("2. Block (%d, %d), Thread (%d, %d): Loaded B_tile 0, i:j(%d, %d)..\n", bx, by, ty, tx, i, j);
        }

        // Synchronize before computation
        __syncthreads();

        /* Accumulate one tile of C from tiles of A and B in shared mem */
        for (int t = 0; t < blockDim.x; t++) {
            // Accumulate for matrix C
            // printf("accu (%.4f), B_tile[%d][%d] = %.2f.\n", accu, t, threadIdx.x, B_tile[t][threadIdx.x]);
            accu += A_tile[threadIdx.y][t] * B_tile[t][threadIdx.x];
            printf("3. Block (%d, %d), Thread (%d, %d): accu += "
                "A_tile[%d][%d] (%.2f) * "
                "B_tile[%d][%d] (%.2f) = %.4f\n", 
                bx, by, ty, tx, 
                threadIdx.y, t, A_tile[threadIdx.y][t], 
                t, threadIdx.x, B_tile[t][threadIdx.x], accu);
        }
        // Synchronize
        __syncthreads();
    }

    if (i < m && j < k) {
        // Store accumulated value to C(i, j)
        C[i * k + j] = accu;

        // Print the computed value for debugging
        printf("4. Block (%d, %d), Thread (%d, %d):Computed C[%d][%d] = %.4f.\n",  bx, by, ty, tx, i, j, C[i * n + j]);
    }
}

// CPU版本的矩阵乘法
void MatrixMulOnHost(int m, int n, int k, float* A, float* B, float* C) {
    for (int Row = 0; Row < m; ++Row) {
        for (int Col = 0; Col < k; ++Col) {
            float sum = 0;
            for (int i = 0; i < n; ++i) {
                float a = A[Row * n + i];
                float b = B[i * k + Col];
                sum += a * b;
            }
            C[Row * k + Col] = sum;
        }
    }
}

void benchmark(int m, int n, int k, int iterations) {
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // 初始化输入矩阵
    for (int i = 0; i < m * k; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < k * n; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 设置网格和块维度
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 测试 MatrixMulKernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "MatrixMulKernel average time: " << milliseconds / iterations << " ms" << std::endl;

    // 测试 MatrixMulKernel_Tiling
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        MatrixMulKernel_Tiling_GPT<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "MatrixMulKernel_Tiling_GPT average time: " << milliseconds / iterations << " ms" << std::endl;

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 主机代码
int main() {
    // int m = 16, n = 16, k = 32; // 矩阵维度
    int m = 4, n = 4, k = 5; // 矩阵维度

    float A[m*n];
    float B[n*k];
    float C_cpu[m*k] = { 0 }; // 结果矩阵C_cpu初始化为0
    float C_gpu[m*k] = { 0 }; // 结果矩阵C_gpu初始化为0

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // 随机生成矩阵 A 的元素
    for (int i = 0; i < m*n; ++i) {
        //A[i] = dis(gen);
        A[i] = static_cast<float>(i+1) / 100.0f; 
    }

    // 随机生成矩阵 B 的元素
    for (int i = 0; i < n*k; ++i) {
        //B[i] = dis(gen);
        B[i] = static_cast<float>(i+1) / 100.0f; 
    }

    // 使用CPU计算矩阵乘法
    MatrixMulOnHost(m, n, k, A, B, C_cpu);

    // 分配GPU内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * k * sizeof(float));
    cudaMalloc((void**)&d_C, m * k * sizeof(float));

    // 数据传输：主机到设备
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // 设置网格和块维度
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH); // 每个块的线程数
    dim3 blocksPerGrid((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("blocksPerGrid size: (%d, %d).\n", blocksPerGrid.x, blocksPerGrid.y);
    printf("threadsPerBlock size: (%d, %d).\n", threadsPerBlock.x, threadsPerBlock.y);
    
    // 调用内核函数
    //MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, d_A, d_B, d_C);
    MatrixMulKernel_Tiling_GPT<<<blocksPerGrid, threadsPerBlock, 2*sizeof(float)*TILE_WIDTH*TILE_WIDTH>>>(m, n, k, d_A, d_B, d_C);

    // 数据传输：设备到主机
    cudaMemcpy(C_gpu, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放GPU内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 比较CPU和GPU的计算结果
    bool result = true;
    for (int i = 0; i < m * k; ++i) {
        if (!isEqualRelative(C_cpu[i], C_gpu[i])) {
            result = false;
            std::cout << "C_cpu["<< i << "] = " << std::fixed << std::setprecision(8) << C_cpu[i] << std::endl;
            std::cout << "C_gpu["<< i << "] = " << std::fixed << std::setprecision(8) << C_gpu[i] << std::endl;
            break;
        }
    }

    if (result) {
        std::cout << "-->CPU and GPU results match!" << std::endl;
    } else {
        std::cout << "-->CPU and GPU results do not match!" << std::endl;
    }

    int m1 = 10240, n1 = 1024, k1 = 1024;
    int iterations = 1000;
    // benchmark(m1, n1, k1, iterations);

    // 打印结果矩阵C_cpu和C_gpu
    std::cout << "CPU result matrix C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            // std::cout << C_cpu[i * k + j] << " ";
            std::cout << std::fixed << std::setprecision(8) << C_cpu[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "GPU result matrix C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            std::cout << std::fixed << std::setprecision(8) << C_gpu[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
