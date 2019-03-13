//
//  FeatureTrack.hpp
//  MySlam
//
//  Created by TuLigen on 2019/3/12.
//

#ifndef FeatureTrack_hpp
#define FeatureTrack_hpp
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
using namespace cv;


typedef std::vector<KeyPoint> KeyPtVector;

typedef std::vector<Point2f>  PtVector;

typedef std::vector<DMatch>   MatchVector;


class FeatureTrack
{
public:
    FeatureTrack()
    {
        mOrb = ORB::create(2000, 1.2f, 8 ,31, 0, 4, ORB::HARRIS_SCORE,31,20);
    }
    
    void calcFeatures(const Mat &img,KeyPtVector &keys, Mat &des)
    {
        mOrb->detect(img, keys);
        mOrb->compute(img, keys, des);
    }
    
    void knn_match(const KeyPtVector &prekeys,const KeyPtVector &curkeys,
                   const Mat &descriptor1,const Mat &descriptor2,
                   PtVector &prepts,PtVector &curpts)
    {
        BFMatcher   mMatcher(NORM_HAMMING);
        const float minRatio = 1.f / 1.2f;
        const int k = 2;
        
        std::vector<std::vector<DMatch> > knnMatches;
        mMatcher.knnMatch(descriptor1, descriptor2, knnMatches, k);
        
        for (size_t i = 0; i < knnMatches.size(); i++) {
            const DMatch& bestMatch = knnMatches[i][0];
            const DMatch& betterMatch = knnMatches[i][1];
            
            float  distanceRatio = bestMatch.distance / betterMatch.distance;
            if (distanceRatio < minRatio)
            {
                prepts.emplace_back(prekeys[bestMatch.queryIdx].pt);
                curpts.emplace_back(curkeys[bestMatch.trainIdx].pt);
            }
        }
    }
    
    void track(const Mat &pre,const Mat &cur,
               const KeyPtVector &prekeys,const KeyPtVector &curkeys,
               Mat &preDes,Mat &curDes,
               PtVector &prepts,PtVector &curpts)
    {
//        BFMatcher   mMatcher(NORM_HAMMING);
//
//        MatchVector matches;
//
//        mMatcher.match(preDes, curDes, matches);
//
//        double min_dist = 10000, max_dist = 0;
//
//        for(size_t i = 0; i < preDes.rows;++i)
//        {
//            double dist = matches[i].distance;
//            if(dist < min_dist)min_dist = dist;
//            if(dist > max_dist)max_dist = dist;
//        }
        
        
//        for(int i = 0; i < matches.size();++i)
//        {
//            if( matches[i].distance <=  max(2 * min_dist,30.0))  //0.2 * (max_dist + min_dist ))
//            {
//
//                Point2f prept = prekeys[matches[i].queryIdx].pt;
//                Point2f curpt = curkeys[matches[i].trainIdx].pt;
//                const int len = 30;//像素距离
//                if( fabs(curpt.x - prept.x) < len &&
//                   fabs(curpt.y - prept.y) < len)
//                {
//                    prepts.emplace_back(prept);
//                    curpts.emplace_back(curpt);
//                }
//            }
//        }
        
        
//        flann::Index flannIndex(preDes,flann::LshIndexParams(12,20,2),cvflann::FLANN_DIST_HAMMING);
//        Mat matchIndex(curDes.rows,2,CV_32SC1),matchDistance(curDes.rows, 2, CV_32FC1);
//        
//        flannIndex.knnSearch(curDes, matchIndex, matchDistance,2, flann::SearchParams());
//        
//        // Lowe's algorithm,获取优秀匹配点
//        for (int i = 0; i < matchDistance.rows; i++)
//        {
//            if (matchDistance.at<float>(i,0) < 0.4 * matchDistance.at<float>(i, 1))
//            {
//                DMatch gd(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
//                
//                prepts.emplace_back(prekeys[gd.queryIdx].pt);
//                curpts.emplace_back(curkeys[gd.trainIdx].pt);
//            }
//        }
//        
        
    }
protected:
    Ptr<ORB> mOrb;
};

#endif /* FeatureTrack_hpp */
