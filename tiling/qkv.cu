#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// 定义查询、键、值的类型
using DataType = float;

// 定义查询、键、值的维度
constexpr int kDim = 64;

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

// 定义 QKV 函数的 CPU 版本
std::vector<DataType> qkv_cpu(const std::vector<DataType>& input) {
    // 输入向量的大小必须是 3 x kDim
    const int input_size = input.size();
    if (input_size != 3 * kDim) {
        throw std::runtime_error("Invalid input size");
    }

    // 分别获取查询、键、值的表示
    const std::vector<DataType> query(input.begin(), input.begin() + kDim);
    const std::vector<DataType> key(input.begin() + kDim, input.begin() + 2 * kDim);
    const std::vector<DataType> value(input.begin() + 2 * kDim, input.end());

    // 计算 QKV 表示
    std::vector<DataType> qkv_output(kDim);
    for (int i = 0; i < kDim; ++i) {
        DataType sum = 0.0;
        for (int j = 0; j < kDim; ++j) {
            sum += query[j] * key[j] * value[j];
        }
        qkv_output[i] = sum / std::sqrt(kDim);
    }

    return qkv_output;
}

// 定义 QKV 函数的 GPU 核函数
__global__ void qkv_gpu_kernel(const DataType* input, DataType* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < kDim) {
        const DataType* query = input;
        const DataType* key = input + kDim;
        const DataType* value = input + 2 * kDim;

        DataType sum = 0.0;
        for (int i = 0; i < kDim; ++i) {
            sum += query[i] * key[i] * value[i];
        }
        output[tid] = sum / sqrtf(kDim);
    }
}

// 定义 QKV 函数的 GPU 版本
std::vector<DataType> qkv_gpu(const std::vector<DataType>& input) {
    // 输入向量的大小必须是 3 x kDim
    const int input_size = input.size();
    if (input_size != 3 * kDim) {
        throw std::runtime_error("Invalid input size");
    }

    // 将输入数据拷贝到 GPU 内存中
    DataType* d_input;
    cudaMalloc((void**)&d_input, input_size * sizeof(DataType));
    cudaMemcpy(d_input, input.data(), input_size * sizeof(DataType), cudaMemcpyHostToDevice);

    // 分配 GPU 内存用于存储输出数据
    DataType* d_output;
    cudaMalloc((void**)&d_output, kDim * sizeof(DataType));

    // 设置 GPU 核函数的线程块和线程数
    const int threads_per_block = 256;
    const int num_blocks = (kDim + threads_per_block - 1) / threads_per_block;

    // 在 GPU 上执行核函数
    qkv_gpu_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output);

    // 将计算结果从 GPU 内存拷贝回主机内存
    std::vector<DataType> output(kDim);
    cudaMemcpy(output.data(), d_output, kDim * sizeof(DataType), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

int main() {
    // 示例输入
    std::vector<DataType> input(3 * kDim, 1.0);

    // CPU 版本 QKV 计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<DataType> output_cpu = qkv_cpu(input);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    // GPU 版本 QKV 计算
    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::vector<DataType> output_gpu = qkv_gpu(input);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;

    // 比较 CPU 和 GPU 计算结果
    bool result = true;
    for (int i = 0; i < kDim; ++i) {
        if (!isEqualRelative(output_cpu[i], output_gpu[i])) {
            result = false;
            break;
        }
    }

    if (result) {
        std::cout << "output_gpu and output_cpu are equal within the tolerance." << std::endl;
    } else {
        std::cout << "output_gpu and output_cpu are not equal within the tolerance." << std::endl;
    }

    // 打印 CPU 版本计算结果和时间
    std::cout << "CPU Version:" << std::endl;
    for (int i = 0; i < kDim; ++i) {
        std::cout << output_cpu[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // 打印 GPU 版本计算结果和时间
    std::cout << "GPU Version:" << std::endl;
    for (int i = 0; i < kDim; ++i) {
        std::cout << output_gpu[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "GPU Time: " << gpu_duration.count() << " ms" << std::endl;

    return 0;
}
