#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }



// __global__ void
// vertical_gray_border(uint8_t * in, uint8_t * out, int size, int rows, int cols)
// {
//     int id = threadIdx.x + blockIdx.x * blockDim.x;
//
//     if (id < rows) {
//         int i = id;
//
//         for (int j = 0; j < cols; ++j) {
//             int current_pixel = i * cols + j;
//             int next_pixel = current_pixel + 1;
//
//             int border = in[next_pixel] - in[current_pixel];
//             out[current_pixel] = border >= 0 ? border : (border * (-1));
//         }
//     }
// }

#define TILE_WIDTH 128
#define TILE_HEIGHT 512


__global__ void
vertical_gray_border(uint8_t * in, uint8_t * out, int size, int rows, int cols)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int block_size = TILE_WIDTH * TILE_HEIGHT;
    int amount_of_blocks = size / block_size;

    int block_number = id / block_size;
    int thread_number = id % block_size;

    int thread_y = thread_number / TILE_WIDTH;
    int thread_x = thread_number % TILE_WIDTH;

    __shared__ char tile_in[TILE_HEIGHT][TILE_WIDTH];
    __shared__ char tile_out[TILE_HEIGHT][TILE_WIDTH];

    if (id < block_size) {
        for (int i = 0; i < amount_of_blocks; ++i) {
            int in_block_index = thread_y * TILE_WIDTH + thread_x;
            int out_block_index = i * block_size + thread_y * TILE_WIDTH + thread_x;

            // tile_in[in_block_index] = in[out_block_index];
            // tile_out[in_block_index] = in[out_block_index + 1] - in[out_block_index];
            // out[out_block_index] =
            //     tile_out[in_block_index] >= 0
            //         ? tile_out[in_block_index]
            //         : (tile_out[in_block_index] * (-1));

            char border = in[out_block_index + 1] - in[out_block_index];
            out[out_block_index] = border >= 0 ? border : (border * (-1));
        }
    }


    // int id = threadIdx.x + blockIdx.x * blockDim.x;
    //
    // if (((block_y * TILE_WIDTH + block_x) * thread_y + thread_x) < size) {
    //     __shared__ int tile_buffer_in[TILE_WIDTH][TILE_WIDTH];
    //     // __shared__ float tile_buffer_out[TILE_WIDTH][TILE_WIDTH];
    //
    //     int row = block_y * blockDim.y + thread_y;
    //     int col = block_x * blockDim.x + thread_x;
    //
    //     // цикл по подматрицам в полосе
    //     for(int p = 0; p < cols / TILE_WIDTH; ++p)
    //     {
    //         tile_buffer_in[thread_y][thread_x] = in[row * cols + (p * TILE_WIDTH + thread_x)];
    //     //     // tile_buffer_out[ty][tx] = b[(p*TILE_WIDTH + ty)*width + col];
    //         __syncthreads();
    //     //
    //     //     // вычисление произведения подматриц s_a и tile_buffer_out
    //         for(int k = 0; k < TILE_WIDTH - 1; ++k) {
    //     //         // int border = tile_buffer_in[ty][k + 1] - tile_buffer_in[ty][k];
    //     //         // out[row * cols + col] = border >= 0 ? border : (border * (-1));
    //             out[] = 127;
    //             __syncthreads();
    //         }
    //     }
    // }
}

int main( int argc, char** argv )
{
    constexpr int width = 2000;
    constexpr int height = 50000;
    Mat image(Size(width, height), CV_8UC1);
    // Mat image= imread("gray_image.jpg", IMREAD_GRAYSCALE);   // Read the file
    //
    // if(!image.data )                                 // Check for invalid input
    // {
    //     cout <<  "Could not open or find the image" << std::endl ;
    //     return -1;
    // }

    randu(image, Scalar(0), Scalar(255));
    // imshow("Original", image);

    int realImageSize = image.channels() * (image.rows - 1) * image.cols;

    uint8_t * cuda_image;
    uint8_t * result_image;

    CHECK(cudaMalloc(&cuda_image, realImageSize * sizeof(uint8_t)));
    CHECK(cudaMalloc(&result_image, realImageSize * sizeof(uint8_t)));
    CHECK(cudaMemcpy(cuda_image, image.data, realImageSize, cudaMemcpyHostToDevice));

    cudaEvent_t startCUDA, stopCUDA;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA,0);

    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    vertical_gray_border<<<height, 512>>>(cuda_image, result_image, realImageSize, image.rows, image.cols);

    cudaEventRecord(stopCUDA,0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());


    float elapsedTimeCUDA;
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cout << "Image rows = " << image.rows << ", cols = " << image.cols << std::endl;
    cout << "CUDA time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << realImageSize / elapsedTimeCUDA /1024 / 1024 / 1.024 << " Gb/s\n";

    CHECK( cudaMemcpy(image.data, result_image,
                      realImageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // imshow("Reversed", image);
    imwrite("reversed.jpg", image);
    waitKey(0);

    return 0;
}
