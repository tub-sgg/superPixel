

#ifndef HTCV_FEATURENOISE_H
#define HTCV_FEATURENOISE_H

#include <string>

#include <opencv2/opencv.hpp>
#include "enum.h"
using namespace std;
using namespace cv;

class FeatureNoise {
public:
    // input: the mean value and std of gaussian additive noise
    FeatureNoise(float mean, float standard_deviation);
    // add gaussian noise to the given image
    Mat addGaussianNoise(const Mat & src);
    // apply different filters to the input image src
    Mat denoise(const Mat & src, filterTypes filter);
private:
    float mean, standard_deviation;

};


#endif //HTCV_FEATURENOISE_H
