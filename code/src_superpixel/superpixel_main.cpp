//
// Created by alxlee on 11.06.20.
//

#include "superPixel.h"
//#if 0
int main(int argc,char** argv){
    /****************************************************************
    ******************start to set the parameters********************
    *****************************************************************
    ****************************************************************/
    //set data images path
    std::string imagepath="../data/image/";
    std::string labelpath="../data/label/";
    std::string uncertaintypath="../data/uncertaintyMap/";
    //set image imdex
    std::vector<int> index={15,23,26,30};
    //set image name pre
    std::string prename="top_mosaic_09cm_area";
    //set confusion matrix
    std::vector<cv::Mat_<float>> confusionMatrixSet;
    // index:15
    cv::Mat_<float> confusionMatrix15=(cv::Mat_<float>(6, 6)<<
    0.620775,     0.10463 ,    0.15263,   0.0265556,   0.0669082,   0.0285015,
    0.125923 ,   0.738295  ,   0.06444,  0.00581489 ,  0.0625603,  0.00296604,
    0.0698345 ,  0.0540798 ,   0.590844 ,   0.244338 ,  0.0346852,  0.00621805,
    0.0100901 , 0.00682819,    0.300401,    0.676473,  0.00546681, 0.000740633,
    0.198144 ,   0.127825  , 0.0466166 , 0.00385052 ,   0.587426,   0.0361381,
    0.0842266,   0.0380064,   0.0158708,  0.00835306,    0.780315 ,  0.0732285);
    //23
    cv::Mat_<float> confusionMatrix23=(cv::Mat_<float>(6, 6)<<
    0.548113  ,  0.122457 ,   0.230646 ,  0.0279947 ,  0.0573977,   0.0133911,
    0.152238 ,   0.705815 ,  0.0772701 , 0.00367114,   0.0593753,  0.00162999,
    0.0497976  , 0.0386349 ,   0.725714 ,    0.16056,    0.022914,  0.00237938,
    0.0109308 , 0.00617157 ,   0.317533 ,   0.660271 ,  0.0044937 ,0.000600837,
    0.214475  , 0.0982715  , 0.0326277  ,0.00958115 ,   0.610151 ,  0.0348935,
    0.352243 ,  0.0982465  , 0.0876741 , 0.00270758  ,  0.378932  ,  0.080196);
    //26
    cv::Mat_<float> confusionMatrix26=(cv::Mat_<float>(6, 6)<<
    0.712137,  0.0661879 , 0.0564434 , 0.0109823 , 0.0838396,  0.0704093,
    0.237699,   0.594148 , 0.0360393 ,0.00412778 ,  0.105892 , 0.0220929,
    0.0900099 , 0.0507953,   0.679984,   0.137287 , 0.0297098,  0.0122142,
    0.0109264 ,0.00352976 ,   0.22059,   0.758447 , 0.0037886, 0.00271825,
    0.228817  , 0.104236 , 0.0501127,  0.0168843 ,  0.467411 ,   0.13254,
    0.0443894 ,0.00862158,  0.0204994 , 0.0105785 , 0.0147092 ,  0.901202);
    //30,
    cv::Mat_<float> confusionMatrix30=(cv::Mat_<float>(6, 6)<<
    0.696115 ,   0.095367 ,  0.0942266,    0.011232 ,  0.0670138,   0.0360454,
    0.161677  ,  0.740787 ,  0.0203581 , 0.00248534 ,  0.0697539 , 0.00493888,
    0.0622871 ,   0.058444 ,   0.630779 ,   0.200554,   0.0440741 , 0.00386116,
    0.00577518 , 0.00402488 ,   0.245699,    0.740668 , 0.00360437, 0.000228459,
    0.167862 ,   0.163855,   0.0841809,   0.0122051,    0.541384,   0.0305127,
    0    ,       0 ,          0     ,      0  ,         0     ,      0);
    confusionMatrixSet.push_back(confusionMatrix15);
    confusionMatrixSet.push_back(confusionMatrix23);
    confusionMatrixSet.push_back(confusionMatrix26);
    confusionMatrixSet.push_back(confusionMatrix30);
    /******************************************************************************
     * ****************************************************************************
     */
     //test
//     auto test=cv::Rect(300,900,100,200);
    for(auto idx:index) {
        //some predefined parameters
        //************read label map data*****************************
        std::string labelMap=labelpath+prename+std::to_string(idx)+".tif";
        cv::Mat src = cv::imread(labelMap, cv::IMREAD_COLOR);
        //************read feature map data*****************************
        std::string featureMap=imagepath+prename+std::to_string(idx)+".tif";
        cv::Mat featureImg = cv::imread(featureMap, cv::IMREAD_COLOR);
        //************ define the labels******************************
        std::list<int> ls = {0, 1, 2, 3, 4, 5};
        //************ set the confusion matrix***********************

        int loc=std::distance(index.begin(),std::find(index.begin(),index.end(),idx));
        cv::Mat_<float> confusionMatrix = confusionMatrixSet[loc];

        //*********** read uncertainty map****************************
        std::string uncertaintyMapName=uncertaintypath+"baselinetile"+std::to_string(idx)+"EntropyGray.png";
        cv::Mat_<cv::Vec3b> uncertainty = cv::imread(uncertaintyMapName);

        //********** define a superpixel class************************
        SuperPixel::superPixel su;

        //convert the data to label image
        cv::Mat image = su.convert_from_color(src);

        //read label map
        su.setImage(image);

        //read feature image
        cv::Mat img = featureImg.clone();
        su.setFeatureImage(img);

        //set super pixel size
        su.setSuperPixelSize(20);

        //set the label list
        su.setLabelList(ls);

        //select the noise rate
        su.setRandomRate(0.75);

        //set the confusion matrix
        su.setConfusionMatix(confusionMatrix);

        //set uncertainty map
        su.setUncertaintyMap(uncertainty);

        //calculate the number of superpixel
        int N;
        int singlePixelsNumber = (int) (img.size().width * img.size().height * su.getRndomRate());
        N = singlePixelsNumber / (su.getSuperixelSize() * su.getSuperixelSize());
        std::cout << " " << N << " SUPER PIXELS WILL BE GENERATED" << std::endl;
        //calculate the Neighborhood matrix(most neighbors

        su.getSuperPixelCoordinates();//calculate coordinates and index
        su.calculateNeighborNumber();// calculate most coordinate matrix
        su.extract_Contour();//calcuate all the contour
        su.getLocalNeighbourMap();
        su.getBoundaryMap();
        /****************************************************************
        ****************End *********************************************
        *****************************************************************
        ****************************************************************/

        //set the out put file name
        std::string superPixelSize=std::to_string(su.getSuperixelSize());
        std::string noiseLevel =std::to_string(su.getRndomRate()*100);

        std::string outputname=std::to_string(idx)+"_"+superPixelSize+"_"+noiseLevel+"%";

        su.superPixelNoise(SuperPixel::RandomSelectionAllRelabel,outputname, N);
        su.superPixelNoise(SuperPixel::UncertaintySelectAllRelabel,outputname, N);
        su.superPixelNoise(SuperPixel::ObjectBorderSelectAllRelabel,outputname, N);

//        su.superPixelNoise(SuperPixel::RandomSelectMostNeighbourRelabel,outputname,N);
//        su.superPixelNoise(SuperPixel::UncertaintyAreaSelectRandomRelabel,outputname,N);
//        su.superPixelNoise(SuperPixel::ObjectBorderSelectRandomRelabel,outputname,N);


        //RELEASE THE CLASS
        su.releaseAllsuperPixel();
    }

}
//#endif


