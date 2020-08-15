
#ifndef HTCV_LABELNOISE_H
#define HTCV_LABELNOISE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "enum.h"
#include <unordered_set>
#include <Eigen/Dense>
#include <random>
using namespace Eigen;
using namespace std;
using namespace cv;



class singleLabelNoise {
public:
    singleLabelNoise(PixelSelectionType selectionType, NewLabelType newLabelType, int number_of_classes,float noiseLevel);

    // add label noise to the given image based on different approaches
    Mat addLabelNoise(const Mat &img, const Mat &labelMap, const MatrixXd &confusionMatrix);

    // select pixels based on uncertainty map or randomly select or from the object boundary
    void sampleSelection(const Mat &uncertainty, const Mat  &labelMap);


private:
    PixelSelectionType selectionType; // define the way to select the pixels
    NewLabelType newLabelType; // define how the relabel selected pixels
    int number_of_classes;
    vector<Point> selected_samples; // stores the position of pixels for future relabelling
    float noiseLevel; // inject how many percentages of label noise to the reference map
};


#endif //HTCV_LABELNOISE_H
