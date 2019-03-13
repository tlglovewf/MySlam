/*
 
 The MIT License
 
 Copyright (c) 2015 Avi Singh
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 
 */

#include "vo_features.h"
#include "FeatureTrack.h"
#include "unistd.h"
#include "Viewer.h"
using namespace cv;
using namespace std;


#define MIN_NUM_FEAT 3000

#define USEWEIYA 1

//光流模式
#define OPATIAL_MODE 0

#if USEWEIYA
//4096 x 2168
#define IMAGEPATH "/Volumes/mac/Data/weiya"
#define REALPOSE  ""
#define MAX_FRAME 150
#else

#define IMAGEPATH "/Volumes/mac/Data/00"
#define REALPOSE  "/Volumes/mac/Data/poses/00.txt"
#define MAX_FRAME 1000
#endif
// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

// 3 x 4 Matrix
// The last column is the translation
//
double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)    {
    
    string line;
    int i = 0;
    ifstream myfile (REALPOSE);
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
        while (( getline (myfile,line) ) && (i<=frame_id))
        {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j=0; j<12; j++)  {
                in >> z ;
                if (j==7) y=z;
                if (j==3)  x=z;
            }
            i++;
        }
        myfile.close();
    }
    
    else {
        cout << "Unable to open file";
        return 0;
    }
    
    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
    
}

double getAbsoluteRT(int frame_id, Mat &R,Mat &t)    {
    
    string line;
    int i = 0;
    ifstream myfile (REALPOSE);
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    R = Mat(3,3,CV_64F);
    if (myfile.is_open())
    {
        while (( getline (myfile,line) ) && (i<=frame_id))
        {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j=0; j<12; j++)  {
                in >> z ;
                if (j==7)
                    y=z;
                else if (j==3)
                    x=z;
                else if (j!=11)
                {
                    int row = j / 4;
                    int col = j % 4;
                    R.at<double>(row,col) = z;
                }
            }
            t = (Mat_<double>(3,1) << x,y,z);
            i++;
        }
        myfile.close();
    }
    
    else {
        cout << "Unable to open file";
        return 0;
    }
    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
}




int main( int argc, char** argv )    {

    Mat img_1, img_2;
    Mat R_f, t_f; //the final rotation and tranlation vectors containing the
    Mat R_r,t_r;//true
    
    ofstream myfile;
    myfile.open ("results1_1.txt");
    
    double scale = 1.00;
    char filename1[200];
    char filename2[200];
#if USEWEIYA
    sprintf(filename1, "%s/0000000%03d_L.jpg",IMAGEPATH, 1);
    sprintf(filename2, "%s/0000000%03d_L.jpg",IMAGEPATH, 2);
#else
    sprintf(filename1, "%s/image_0/%06d.png",IMAGEPATH, 0);
    sprintf(filename2, "%s/image_0/%06d.png",IMAGEPATH, 1);
#endif
    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);
    
    //read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);
    
    if ( !img_1_c.data || !img_2_c.data ) {
        std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }
    
    // feature detection, tracking
    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    
#if OPATIAL_MODE
    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);
    
    featureDetection(img_1, points1);        //detect features in img_1
    vector<uchar> status;
    featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2
#else
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);
    
    FeatureTrack track;
    KeyPtVector prekeys;
    KeyPtVector curkeys;
    Mat preDes;
    Mat curDes;
    track.calcFeatures(img_1, prekeys, preDes);
    track.calcFeatures(img_2, curkeys, curDes);
    
    //track.track(img_1, img_2, prekeys, curkeys, preDes, curDes, points1, points2);
    
    track.knn_match(prekeys, curkeys, preDes, curDes, points1, points2);
    
    assert(points1.size() == points2.size());
    assert(points1.size() > 0);
    preDes  = curDes.clone();
    prekeys.swap(curkeys);
    
#endif
    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    
#if USEWEIYA
//    2.3695365586649123e+03  0. 2.0443736115794320e+03 0.
//    2.3695365586649123e+03 1.0750972331437883e+03 0. 0. 1.
    double focal = 2.3695365586649123e+03;
    cv::Point2d pp(2.0443736115794320e+03, 1.0750972331437883e+03);
#else
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);
#endif
    //recovering the pose and the essential matrix
    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);
    
    Mat prevImage = img_2;
    Mat currImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;
    
    char filename[100];
    
    R_f = R.clone();
    t_f = t.clone();
    
#if USEWEIYA
#else
    getAbsoluteRT(0, R_r, t_r);
#endif
    
    clock_t begin = clock();
    
    namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
    namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.
    
#define WIN_SIZE 1000
    Mat traj = Mat::zeros(WIN_SIZE, WIN_SIZE, CV_8UC3);
    
    for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)    {
        
#if USEWEIYA
        sprintf(filename, "%s/0000000%03d_L.jpg",IMAGEPATH,numFrame+1);
#else
        sprintf(filename, "%s/image_0/%06d.png",IMAGEPATH, numFrame);
#endif
        //cout << numFrame << endl;
        Mat currImage_c = imread(filename);
        
#if OPATIAL_MODE
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
#else
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        track.calcFeatures(currImage, curkeys, curDes);
        prevFeatures.clear();
        currFeatures.clear();
//        track.track(prevImage, currImage, prekeys, curkeys, preDes, curDes, prevFeatures, currFeatures);
        track.knn_match(prekeys, curkeys, preDes, curDes, prevFeatures, currFeatures);
        prekeys.swap(curkeys);
        preDes  = curDes.clone();
        assert(prevFeatures.size() == currFeatures.size());
        cout << prevFeatures.size() << endl;
#endif
        E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
        
        Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);
        
        
        for(int i=0;i<prevFeatures.size();i++)    {   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
            prevPts.at<double>(0,i) = prevFeatures.at(i).x;
            prevPts.at<double>(1,i) = prevFeatures.at(i).y;
            
            currPts.at<double>(0,i) = currFeatures.at(i).x;
            currPts.at<double>(1,i) = currFeatures.at(i).y;
        }
        
#if USEWEIYA
        scale = 4.0;
#else
        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
        
#endif
        
#if USEWEIYA
#else
        Mat R_,t_;
        getAbsoluteRT(numFrame, R_, t_);
#endif
        
        //      std::cout << R << endl << t << std::endl;
        
//        cout << t << endl;
        if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            
            t_f = t_f + scale*(R_f*t);
            R_f = R * R_f;
            
#if USEWEIYA
#else
            //true
            t_r = R_r * t_;
#endif
        }
        
        else {
            //cout << "scale below 0.1, or incorrect translation" << endl;
        }
        
        // lines for printing results
        // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;
        
        // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT)    {
            //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
            //cout << "trigerring redection" << endl;
#if OPATIAL_MODE
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);
#else
            
#endif
            
        }
        
        prevImage = currImage.clone();
        prevFeatures = currFeatures;
        
    
        
        const int half_size = WIN_SIZE >> 1;
#if USEWEIYA
        int x = int(t_f.at<double>(0)) + half_size;
        int y = int(t_f.at<double>(2)) + 100;
        circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);//红色计算轨迹
#else
        int x = int(t_f.at<double>(0)) + half_size;
        int y = int(t_f.at<double>(2)) + 100;
        circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 1);//红色计算轨迹
        
        x = int(t_r.at<double>(0)) + half_size ;
        y = int(t_r.at<double>(2)) + 100;
        circle(traj, Point(x, y) ,1, CV_RGB(0,255,0), 1);//绿色真实轨迹
#endif
        rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
        
        
#if USEWEIYA
        
        Mat wy = currImage_c.clone();
        resize(wy, wy, Size(wy.cols >> 2,wy.rows >> 2));
        
        imshow( "Road facing camera", wy );
        
#else
        imshow( "Road facing camera", currImage_c );
        
#endif
        imshow( "Trajectory", traj );
        
        waitKey(1);
    }
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl;
    waitKey(0);
    return 0;
}


