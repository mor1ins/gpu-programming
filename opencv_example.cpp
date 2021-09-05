#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>    // std::swap
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main( int argc, char** argv )
{
    Mat image;
    image = imread("pic.png", IMREAD_COLOR);   // Read the file
    if(! image.data )                                 // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    auto start = high_resolution_clock::now();

    for(int i = 0; i < image.rows / 2; i++)
    {
        //pointer to 1st pixel in row
        Vec3b* p_head = image.ptr<Vec3b>(i);
        Vec3b* p_tail = image.ptr<Vec3b>(image.rows - 1 - i);
        for (int j = 0; j < image.cols; j++) {
            swap(p_head[j], p_tail[j]);
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;

    imwrite("pic2.jpg",image);

    //show image
    namedWindow( "Display window", WINDOW_AUTOSIZE );    // Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
