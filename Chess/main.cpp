//:completeSettings = none
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat frame(630,891,CV_8U); // = imread( "3A4.bmp"); // cols*rows = 630*891
    
    int nc = frame.channels();
    
    int nWidthOfROI = 90;
    
    for (int j=0;j<frame.rows;j++)
    {
        uchar* data= frame.ptr<uchar>(j);
        for(int i=0;i<frame.cols*nc;i+=nc)
        {
            if( (i/nc/nWidthOfROI + j/nWidthOfROI) % 2)
            {
                // bgr
                data[i/nc*nc + 0] = 255 ;
                data[i/nc*nc + 1] = 255 ;
                data[i/nc*nc + 2] = 255 ;
            }
        }
    }
    
//    imshow("test",frame);
    
//    waitKey(0);
    imwrite("/Users/TLG/Downloads/a4chess.jpg", frame);
    return 0;
}

