#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>

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



__global__ void horizontal_refrection(uint8_t * in, uint8_t * out, int size,
                                      int rows, int cols, int channels)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < size) {
        int count_of_pixels = id / channels;

        int i = count_of_pixels / cols;
        int j = id % cols;
        int k = id % channels;

        int const_part = j * channels + k;
        int cols_with_channels = cols * channels;

        int output_index = i * cols_with_channels + const_part;
        int input_index = (rows - 1 - i) * cols_with_channels + const_part;

        out[output_index] = in[input_index];
    }

}

int main( int argc, char** argv )
{
    constexpr long long width = 800;
    constexpr long long height = 570;
    constexpr long long channels = 3;
    constexpr long long imageSize = width * height * channels;
    Mat image;
    image = imread("pic.jpg", IMREAD_COLOR);   // Read the file

    if(!image.data )                                 // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    imshow("Original", image);

    int realImageSize = image.channels() * image.rows * image.cols;
    if (realImageSize != imageSize) {
        cout <<  "Compile-time buffer size doesn't match with real size" << std::endl ;
        return -1;
    }

    uint8_t * cuda_image;
    uint8_t * result_image;

    CHECK(cudaMalloc(&cuda_image, imageSize * sizeof(uint8_t)));
    CHECK(cudaMalloc(&result_image, imageSize * sizeof(uint8_t)));
    CHECK(cudaMemcpy(cuda_image, image.data, imageSize, cudaMemcpyHostToDevice));

    auto start = high_resolution_clock::now();

    horizontal_refrection<<<1000000, 512>>>(cuda_image, result_image, imageSize,
                                            height, width, channels);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;

    CHECK( cudaMemcpy(image.data, result_image, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    imshow("Reversed", image);
    waitKey(0);

    return 0;
}
