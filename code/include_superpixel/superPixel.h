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

        static cv::Mat convert_from_color(const cv::Mat &src);

        static cv::Mat convert_to_color(const cv::Mat& src);

        int getSuperixelSize()const {return superPixelSize;}

        float getRndomRate()const{return randomRate;}

        void releaseAllsuperPixel(){ allsuperPixelCoordinates.clear();}

    private:
        PixelSelectionType selectionType;
        NewLabelType addLabelNoiseType;
        std::vector<std::vector<cv::Point>> allsuperPixelCoordinates;
        int MaximumSuperPixelIndex;
        cv::Mat_<int> NeighborMatrix;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<int> selectedSuperPixelIndex;
        float randomRate;
        cv::Mat image;//index map
        int superPixelSize;
        std::list<LABEL> ls;
        cv::Mat_<cv::Vec3b> uncertainty;
        cv::Mat addedLabelNoiseImage;
        cv::Mat_<float> confusionMatrix;
        cv::Mat featureImg;
        std::vector<LABEL> localneighborMap;
        cv::Mat contourMap;
        cv::Mat labelMap;
        std::vector<std::vector<int>> superpixelRelationMap;
    public:
        void extract_Contour();

        void getSuperPixelImage(cv::Mat &mask,int &labelNumbers);

        void getSuperPixelCoordinates();

         LABEL preprocessSuperPixel(const std::vector<cv::Point> &superpixel);

        //how often the neighborhood 2nd traverse the whoe image and test
        void calculateNeighborNumber();

        void getLocalNeighbourMap();

        void getBoundaryMap();

        void calculateSuperpixelRelationshipMap();

    public:
        void sampleSelection(int N=100);//N define the size of superpixel required

        void addLabelNoise();

        void superPixelNoise(SuperPixel::options option,std::string& outputFilenameIndex,int N=100);

        void saveColorImage(const std::string& filename);

    };

}
#endif //NOISE_SUPERPIXEL_H
