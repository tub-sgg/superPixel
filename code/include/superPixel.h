//
// Created by alxlee on 05.07.20.
//

#ifndef NOISE_SUPERPIXEL_H
#define NOISE_SUPERPIXEL_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>
#include <random>
#include <algorithm>
#include <list>
#include <map>
#include <climits>
#include <numeric>
#include <ctime>
#include "omp.h"


#include "super_pixel_label_noise.h"

typedef int LABEL ;


namespace SuperPixel {

    class superPixel {
    public:
        //default construction
        superPixel()=default;

        void setImage(cv::Mat& src){image=src;}

        void setConfusionMatix(cv::Mat_<float> &confusion) { confusionMatrix = confusion; }

        void setRandomRate(const float &rate) { randomRate = rate; }

        void setSuperPixelSize(const int &superpixelsize) { superPixelSize = superpixelsize; }

        void setLabelList(const std::list<LABEL> &label) { ls = label; }

        void setUncertaintyMap(const cv::Mat_<cv::Vec3b> &uncertaintyImage) { uncertainty = uncertaintyImage; }

        void setPixelSelectionType(PixelSelectionType type) { selectionType = type; }

        void setAddLabelNoiseType(NewLabelType type) { addLabelNoiseType = type; }

        void setFeatureImage(cv::Mat &src){featureImg=src.clone();}

        cv::Mat getAddedLabelNoiseImage() { return addedLabelNoiseImage; }

        static cv::Mat convert_from_color(const cv::Mat &src);

        static cv::Mat convert_to_color(const cv::Mat& src);

        int getSuperixelSize()const {return superPixelSize;}

//        int getPixelNumberinContours()const;
//
//        int getPixelNumberinUncertaintyArea() const;

        float getRndomRate()const{return randomRate;}

        void releaseAllsuperPixel(){ allsuperPixelCoordinates.clear();}

    private:
        PixelSelectionType selectionType;
        NewLabelType addLabelNoiseType;
        std::vector<std::vector<cv::Point>> allsuperPixelCoordinates;
        int MaximumSuperPixelIndex;
        cv::Mat_<int> NeighborMatrix;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<std::vector<cv::Point>> selectedSuperPixel;
        std::vector<int> selectedSuperPixelIndex;
        float randomRate;
        cv::Mat image;//index map
        int superPixelSize;
        std::list<LABEL> ls;
        cv::Mat_<cv::Vec3b> uncertainty;
        cv::Mat addedLabelNoiseImage;
        cv::Mat_<float> confusionMatrix;
        cv::Mat featureImg;
        cv::Mat superpixelImage;
        cv::Mat localSuperpixelNeighbormap;
        int width,  height;//superpixelimage
        std::vector<LABEL> localneighborMap;
        cv::Mat contourMap;
        cv::Mat labelMap;
        std::vector<std::vector<int>> superpixelRelationMap;
    public:

//        int extract_Contour(
//                std::map<LABEL, std::vector<std::vector<cv::Point>>> &contours,
//                std::list<std::vector<cv::Vec4i>> &hierarchies);

        void extract_Contour();

//        static bool testPointInPolygon(const cv::Point &point,
//                                       const std::vector<cv::Point> &contour);


        void getSuperPixelImage(cv::Mat &mask,int &labelNumbers);

        void getSuperPixelCoordinates();

//        static bool testIntersectionSuperPixel(const std::map<LABEL, std::vector<std::vector<cv::Point>>> &contours,
//                                               const std::vector<cv::Point> &superPixels//coordinates
//        );
//        static bool testIntersectionSuperPixel(const std::vector<cv::Point> &contour,
//                                               const std::vector<cv::Point> &superPixels);

//        bool
//        testSuperPixelIntersectionReturnContour(const std::map<LABEL, std::vector<std::vector<cv::Point>>> &contours,
//                                                const std::vector<cv::Point> &superPixel,//coordinates
//                                                std::multimap<LABEL, std::vector<cv::Point>> &result//corresponding contours
//        );
//
//        bool
//        testSuperPixelIntersectionReturnContour(const std::vector<std::vector<cv::Point>> &contours,
//                                                const std::vector<cv::Point> &superPixels,
//                                                std::vector<std::vector<cv::Point>>   &result
//                                                            );
         LABEL preprocessSuperPixel(const std::vector<cv::Point> &superpixel);

        //how often the neighborhood 2nd traverse the whoe image and test
        void calculateNeighborNumber();


//        static int miniColSuperPixel(std::vector<cv::Point>& superPixel);//find the col position
//        static int minRowwSuperPixel(std::vector<cv::Point>& superPixel);// find the label correspondingly
//        void createSuperPixelImage();
//        void calculateLocalNeighborMap();
        void getLocalNeighbourMap();
        void getBoundaryMap();
//        bool testsuperpixeltouchanothersuperpixel(int index,std::vector<int> &newvector);
        void calculateSuperpixelRelationshipMap();

    public:
        void sampleSelection(int N=100);//N define the size of superpixel required

        void addLabelNoise();

        void superPixelNoise(SuperPixel::options option,std::string& outputFilenameIndex,int N=100);

        void saveColorImage(const std::string& filename);

    };

}
#endif //NOISE_SUPERPIXEL_H
