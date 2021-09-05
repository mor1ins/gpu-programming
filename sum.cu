#include <iostream>

using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

// размер массива ограничен максимальным размером пространства потоков
__global__ void sum_simple(float *a, float *b, float *c, int N)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N)
        c[i] = a[i]+b[i];
}

// работает даже для очень больших массивов
__global__ void sum_universal(float *a, float *b, float *c, int N)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int threadsNum = blockDim.x*gridDim.x;
    for (int i = id; i < N; i+=threadsNum)
        c[i] = a[i]+b[i];
}

int main(void)
{
    int N = 10*1000*1000;
    float *host_a, *host_b, *host_c, *host_c_check;
    float *dev_a, *dev_b, *dev_c;

    cudaEvent_t startCUDA, stopCUDA;
    clock_t startCPU;
    float elapsedTimeCUDA, elapsedTimeCPU;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);
    host_a = new float[N];
    host_b = new float[N];
    host_c = new float[N];
    host_c_check = new float[N];
    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
        host_b[i] = 2*i;
    }
    startCPU = clock();

//#pragma omp parallel for
    for (int i = 0; i < N; i++) host_c_check[i] = host_a[i] + host_b[i];
    elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
    cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
    cout << "CPU memory throughput = " << 3*N*sizeof(float)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";

    CHECK( cudaMalloc(&dev_a, N*sizeof(float)) );
    CHECK( cudaMalloc(&dev_b, N*sizeof(float)) );
    CHECK( cudaMalloc(&dev_c, N*sizeof(float)) );
    CHECK( cudaMemcpy(dev_a, host_a, N*sizeof(float),cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(dev_b, host_b, N*sizeof(float),cudaMemcpyHostToDevice) );

    cudaEventRecord(startCUDA,0);

    // размер массива ограничен максимальным размером пространства потоков
    sum_simple<<<(N+511)/512, 512>>>(dev_a, dev_b, dev_c, N);

    // работает даже для очень больших массивов
    //sum_universal<<<100, 512>>>(dev_a, dev_b, dev_c, N);

    cudaEventRecord(stopCUDA,0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << 3*N*sizeof(float)/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";

    CHECK( cudaMemcpy(host_c, dev_c, N*sizeof(float),cudaMemcpyDeviceToHost) );

    // check
    for (int i = 0; i < N; i++)
        if (abs(host_c[i] - host_c_check[i]) > 1e-6)
        {
            cout << "Error in element N " << i << ": c[i] = " << host_c[i]
                 << " c_check[i] = " << host_c_check[i] << "\n";
            exit(1);
        }
    CHECK( cudaFree(dev_a) );
    CHECK( cudaFree(dev_b) );
    CHECK( cudaFree(dev_c) );
    return 0;
}
