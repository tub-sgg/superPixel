//
// Created by alxlee on 11.06.20.
//

#include "superPixel.h"

int main(int argc,char** argv){
    /****************************************************************
    ******************start to set the parameters********************
    *****************************************************************
    ****************************************************************/
    //some predefined parameters
    //************read label map data*****************************
    cv::Mat src=cv::imread("../data/top_mosaic_09cm_area23.tif");
    //************ define the labels******************************
    std::list<int> ls={0,1,2,3,4,5};
    //************ set the confusion matrix***********************
    cv::Mat_<float> confusionMatrix=(cv::Mat_<float>(6, 6)<<
    0.687066 , 0.0480243  ,0.0652707,  0.0119606  , 0.106975 , 0.0807033,
    0.253501  ,  0.54205 ,  0.039177, 0.00491381,   0.135552 , 0.0248065,
    0.0771283 , 0.0382524 ,   0.69286 ,  0.149886 , 0.0312597,  0.0106145,
    0.00527714 ,0.00284537 ,  0.185063  , 0.801442, 0.00250271 ,0.00287024,
    0.183743,  0.0845672 , 0.0587056,  0.0182974,   0.513253,   0.141434,
    0.0273497, 0.00847397,   0.026936 , 0.0104175,   0.018386,   0.908437);



    //*********** read uncertainty map****************************
    cv::Mat uncertainty=cv::imread("../data/tile23EntropyGray.png");

    //********** define a superpixel class************************
    SuperPixel::superPixel su;

    //convert the data to label image
    cv::Mat image=su.convert_from_color(src(cv::Rect(0,0,500,500)));

    //read label map
    su.setImage(image);

    //set super pixel size
    su.setSuperPixelSize(20);
    //set the label list
    su.setLabelList(ls);

    //select the random rate
    su.setRandomRate(0.1);

    //set the confusion matrix
    su.setConfusionMatix(confusionMatrix);

    //set uncertainty map
    su.setUncertaintyMap(uncertainty(cv::Rect(0,0,500,500)));

    //set Uncertainty Threshold
    su.setUncertaintyThreshold(0.75);

    /****************************************************************
    ****************End *********************************************
    *****************************************************************
    ****************************************************************/

//    //random select , random relabel
//    su.superPixelNoise(SuperPixel::options::RandomSelectRandomRelabel);
//    //save data
//    su.saveColorImage("RandomSelectRandomRelabel");

//    //random select, confusion relabel
//    su.superPixelNoise(SuperPixel::options::RandomSelectConfusionRelabel);
//    //save data
//    su.saveColorImage("RandomSelectConfusionRelabel");

//    //random select, how often neighbors
//    su.superPixelNoise(SuperPixel::options::RandomSelectNeighbourInReferenceRelabel);
//    //save data
//    su.saveColorImage("RandomSelectNeighbourInReferenceRelabel");

//    //random select, nearesst neighbors
//    su.superPixelNoise(SuperPixel::options::RandomSelectNeighbourInFeatureSpaceRelabel);
//    //save data
//    su.saveColorImage("RandomSelectNeighbourInFeatureSpaceRelabel");

//    //uncertainty area select, random neighbors
//    su.superPixelNoise(SuperPixel::options::UncertaintyAreaSelectRandomRelabel);
//    //save data
//    su.saveColorImage("UncertaintyAreaSelectRandomRelabel");
//
//    //uncertainty area select, confusion
//    su.superPixelNoise(SuperPixel::options::UncertaintyAreaSelectConfusionRelabel);
//    //save data
//    su.saveColorImage("UncertaintyAreaSelectConfusionRelabel");
//
//    //uncertainty area select, how often
//    su.superPixelNoise(SuperPixel::options::UncertaintyAreaSelectNeighbourInReferenceRelabel);
//    //save data
//    su.saveColorImage("UncertaintyAreaSelectNeighbourInReferenceRelabel");
//
//    uncertainty area select, nearest
//    su.superPixelNoise(SuperPixel::options::UncertaintyAreaSelectNeighbourInFeatureSpaceRelabel);
//    //save data
//    su.saveColorImage("UncertaintyAreaSelectNeighbourInFeatureSpaceRelabel");


    //boundary  select, random neighbors
    su.superPixelNoise(SuperPixel::options::ObjectBorderSelectRandomRelabel);
    //save data
    su.saveColorImage("ObjectBorderSelectRandomRelabel");

//    //boundary  select select, confusion
//    su.superPixelNoise(SuperPixel::options::ObjectBorderSelectConfusionRelabel);
//    //save data
//    su.saveColorImage("ObjectBorderSelectConfusionRelabel");
//
//    //boundary  select select, how often
//    su.superPixelNoise(SuperPixel::options::ObjectBorderSelectNeighbourInReferenceRelabel);
//    //save data
//    su.saveColorImage("ObjectBorderSelectNeighbourInReferenceRelabel");
//
//    //boundary  select select, nearest
//    su.superPixelNoise(SuperPixel::options::ObjectBorderSelectNeighbourInFeatureSpaceRelabel);
//    //save data
//    su.saveColorImage("ObjectBorderSelectNeighbourInFeatureSpaceRelabel");



}