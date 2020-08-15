
#include "FeatureNoise.h"

#include "opencv2/xphoto.hpp"
using namespace cv;
using namespace std;


// class initialization: the mean value and std of gaussian additive noise
FeatureNoise::FeatureNoise(float mean, float standard_deviation):mean(mean), standard_deviation(standard_deviation) {

}

// add gaussian noise to the given image

Mat FeatureNoise::addGaussianNoise(const Mat &src) {
    Mat dst(src.size(), src.type());
    if(src.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return dst;
    }
    Mat src_16SC;
    Mat Gaussian_noise = Mat(src.size(),CV_16SC3);
    randn(Gaussian_noise,Scalar::all(mean), Scalar::all(standard_deviation));

    src.convertTo(src_16SC,CV_16SC3);
    addWeighted(src_16SC, 1.0, Gaussian_noise, 1.0, 0.0, src_16SC);
    src_16SC.convertTo(dst,src.type());
    return dst;
}

// apply different filters to the input image src for reducing feature noise
Mat FeatureNoise::denoise(const Mat &src, filterTypes filter) {
    Mat dst;
    if(filter == bilateral_filter){
        bilateralFilter(src, dst, 5, 20, 20);
        return dst;
    }else if(filter == NLM_filter){
        fastNlMeansDenoising(src, dst, 7, 7, 21);
        return dst;
    }
    else if(filter == BM3D_filter){
        xphoto::bm3dDenoising(src, dst);
        return dst;
    }
    else{
        cerr<<"please input right filter type for reducing feature noise"<<endl;
        return dst;
    }

}
