

#include "InputOutput.h"

using namespace cv;

InputOutput::InputOutput(string image_folder, string label_folder) : image_folder(image_folder), label_folder(label_folder) {}


// given image indices, load the images and stores them in a vector of mat
vector<Mat> InputOutput::loadImages(const vector<int> &imageIndices) {
    vector<Mat> all_images;
    for (const auto &index: imageIndices) {
        Mat img = imread(image_folder + "top_mosaic_09cm_area" + to_string(index) + ".tif");
        all_images.push_back(img);
    }
    return all_images;
}
// given image indices, load the corresponding reference maps and stores them in a vector of mat
vector<Mat> InputOutput::loadLabelMatrices(const vector<int> &imageIndices) {
    vector<Mat> label_maps;

    for (const auto &index: imageIndices) {
        Mat label_color_coded = imread(label_folder + "top_mosaic_09cm_area" + to_string(index) + ".tif");
        label_maps.push_back(convert_from_color(label_color_coded));
    }
    return label_maps;
}

Mat InputOutput::convert_from_color(const Mat &src) {
    Mat label_map = Mat::zeros(src.size(), CV_8U);
    for (int i = 0; i < label_map.size().height; i++) {
        for (int j = 0; j < label_map.size().width; j++) {
            if (src.at<Vec3b>(i, j) == Vec3b(255, 255, 255)) label_map.at<uchar>(i, j) = 0;
            if (src.at<Vec3b>(i, j) == Vec3b(255, 0, 0)) label_map.at<uchar>(i, j) = 1;
            if (src.at<Vec3b>(i, j) == Vec3b(255, 255, 0)) label_map.at<uchar>(i, j) = 2;
            if (src.at<Vec3b>(i, j) == Vec3b(0, 255, 0)) label_map.at<uchar>(i, j) = 3;
            if (src.at<Vec3b>(i, j) == Vec3b(0, 255, 255)) label_map.at<uchar>(i, j) = 4;
            if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 255)) label_map.at<uchar>(i, j) = 5;
        }
    }
    return label_map;
}

Mat InputOutput::convert_to_color(const Mat & src){

    Mat label_map_color_coded = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < label_map_color_coded.size().height; i++) {
        for (int j = 0; j < label_map_color_coded.size().width; j++) {
            if (src.at<uchar>(i, j) == 0) label_map_color_coded.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            if (src.at<uchar>(i, j) == 1) label_map_color_coded.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
            if (src.at<uchar>(i, j) == 2) label_map_color_coded.at<Vec3b>(i, j) = Vec3b(255, 255, 0);
            if (src.at<uchar>(i, j) == 3) label_map_color_coded.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
            if (src.at<uchar>(i, j) == 4) label_map_color_coded.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
            if (src.at<uchar>(i, j) == 5) label_map_color_coded.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
        }
    }
    return label_map_color_coded;
}