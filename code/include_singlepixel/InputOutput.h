

#ifndef HTCV_INPUTOUTPUT_H
#define HTCV_INPUTOUTPUT_H

// define the functions to load images and corresponding reference maps from given disk

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
class InputOutput {
public:
    InputOutput(string image_folder, string label_folder);
    vector<Mat> loadImages(const vector<int>& imageIndices);
    vector<Mat> loadLabelMatrices(const vector<int> & imageIndices);
    // convert label matrix to color-coded reference map
    Mat convert_from_color(const Mat &src);
    Mat convert_to_color(const Mat & src);
private:
    string image_folder, label_folder;
};



#endif //HTCV_INPUTOUTPUT_H
