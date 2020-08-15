#include <iostream>
#include <experimental/filesystem>
#include <Eigen/Dense>
#include "ImgIndex.h"
#include "enum.h"
#include "InputOutput.h"
#include "FeatureNoise.h"
#include "singleLabelNoise.h"

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem;
int main() {

    string Folder= "../data";
    vector<string> classes = {"roads", "buildings", "low veg", "trees", "cars", "clutter"};
    int number_of_classes = classes.size();

    string image_folder = Folder + "/image/";
    string label_folder = Folder + "/label/";



//    //***********************************feature noise **************************************
//    string featureNoiseFolder = Folder+"/top_featureNoise/";
//    string featureNoiseReduction = Folder+"/top_featureNoiseReduction/";
//    InputOutput inputOutput(image_folder, label_folder);
//    vector<int> all_images_indices = {1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37};
//    vector<Mat> allImages = inputOutput.loadImages(all_images_indices);
//
//
//    FeatureNoise featureNoise(0, 10);
//    for(int i=0;i<all_images_indices.size();i++){
//        Mat noisy_image = featureNoise.addGaussianNoise(allImages[i]);
//        if(!fs::v1::exists(featureNoiseFolder)){
//            fs::v1::create_directories(featureNoiseFolder);
//        }
//        imwrite(featureNoiseFolder+"top_mosaic_09cm_area" + to_string(all_images_indices[i]) + ".tif", noisy_image);
//        Mat denoised_image = featureNoise.denoise(noisy_image, NLM_filter); // bilateral_filter, NLM_filter, BM3D_filter
//        if(!fs::v1::exists(featureNoiseReduction)){
//            fs::v1::create_directories(featureNoiseReduction);
//        }
//        imwrite(featureNoiseReduction+"top_mosaic_09cm_area" + to_string(all_images_indices[i]) + ".tif", denoised_image);
//
//    }



//    ***********************************label noise **************************************
//  randomSelection, uncertaintyArea, objectBorder
//  randomRelabel, CMrelabel, neighbourInReference, neighbourInFeatureSpace
    string outputFolder = Folder+"/labelNoise/objectBorder_globalNeighbourInReference/";
    if(!fs::v1::exists(outputFolder)){
        fs::v1::create_directories(outputFolder);
    }
    MatrixXd confusionMatrix = MatrixXd::Zero(number_of_classes, number_of_classes);
    confusionMatrix <<0.66271  , 0.0914154,    0.114901,   0.0168049  , 0.0712741 ,  0.0428943,
    0.176584 ,   0.686265  , 0.0457752 , 0.00398047 ,  0.0779473,  0.00944831,
    0.0639278  ,  0.049949   , 0.654271  ,  0.193933 ,  0.0328179  ,0.00510117,
    0.00967551 , 0.00543313  ,  0.279732  ,  0.699713  ,0.00446814 ,0.000978163,
    0.20133    , 0.12885 ,  0.0608282,   0.0128487 ,   0.520989  , 0.0751538,
    0.0501761   ,0.0105827  ,  0.021555 ,  0.0104118 ,  0.0327243  ,   0.87455;

    PixelSelectionType selectionType = objectBorder;  //randomSelection, uncertaintyArea, objectBorder
    NewLabelType newLabelType = globalNeighbourInReference; //randomRelabel, CMrelabel, localNeighbourInReference,
                                            // globalNeighbourInReference, neighbourInFeatureSpace
    vector<int> images_index = { 15, 23, 26, 30};

    InputOutput inputOutput(image_folder, label_folder);
    vector<Mat> all_label_maps = inputOutput.loadLabelMatrices(images_index);
    vector<Mat> all_images = inputOutput.loadImages(images_index);
    vector<Mat> uncertaintyMaps;
    for(auto&index:images_index){
        Mat uncertainty = imread(Folder+"/uncertaintyMap/"+"baselineknntile"+to_string(index)+"EntropyGray.png",0);
        uncertaintyMaps.push_back(uncertainty);
    }

    for(int i=0;i<images_index.size();i++){
        singleLabelNoise labelNoise(selectionType, newLabelType, number_of_classes, 0.1);
        labelNoise.sampleSelection(uncertaintyMaps[i], inputOutput.convert_to_color(all_label_maps[i]));
        Mat newlabelMap = labelNoise.addLabelNoise(all_images[i], all_label_maps[i], confusionMatrix);
        imwrite(outputFolder+"top_mosaic_09cm_area" + to_string(images_index[i]) + ".tif", inputOutput.convert_to_color(newlabelMap));
//        imwrite("tile_"+to_string(images_index[i])+"_newLabelMap.png", inputOutput.convert_to_color(newlabelMap));
    }




////  ***********************************KNN classifier*************************************
//    vector<string> labelNoiseType={"randomSelection_CMrelabel", "randomSelection_globalNeighbourInReference",
//                                 "randomSelection_localNeighbourInReference", "randomSelection_neighbourInFeatureSpace",
//                                   "objectBorder_CMrelabel", "objectBorder_globalNeighbourInReference",
//                                   "objectBorder_localNeighbourInReference", "objectBorder_neighbourInFeatureSpace",
//                                   "objectBorder_randomRelabel", "uncertaintyArea_CMrelabel",
//                                   "uncertaintyArea_globalNeighbourInReference", "uncertaintyArea_localNeighbourInReference",
//                                   "uncertaintyArea_neighbourInFeatureSpace","uncertaintyArea_randomRelabel"};
//
////    vector<string> labelNoiseType={"randomSelection_CMrelabel", "randomSelection_globalNeighbourInReference",
////                                   "randomSelection_localNeighbourInReference", "randomSelection_neighbourInFeatureSpace"};
//
//    for(const auto &noise: labelNoiseType){
//        string image_folder = Folder + "/top/";
//        string train_label_folder = Folder + "/labelNoise/"+noise+"/";
//        string test_label_folder =Folder + "/gts_for_participants/";
//        InputOutput trainInputOutput(image_folder, train_label_folder);
//        vector<int> train_images_index = {23, 26, 30};
//        vector<Mat> train_images = trainInputOutput.loadImages(train_images_index);
//        // integer label map
//        vector<Mat> train_label_maps = trainInputOutput.loadLabelMatrices(train_images_index);
//        int number_of_samples_in_each_class = 2000;
//        int patch_size = 3;
//        int K = 11;
//
//        InputOutput testInputOutput(image_folder, test_label_folder);
//        vector<int> test_images_index = {15};
//        vector<Mat> test_images = testInputOutput.loadImages(test_images_index);
//        vector<Mat> test_label_maps = testInputOutput.loadLabelMatrices(test_images_index);
//
//        KNN knn(K, patch_size, classes, number_of_samples_in_each_class);
//        knn.train(train_images, train_label_maps);
//        vector<Mat> predicted_label_maps = knn.predict(test_images, test_label_maps,test_images_index,noise);
//        for(int i=0;i<test_images.size();i++){
//            if(!fs::v1::exists("../knn_prediction/"+noise)){
//                fs::v1::create_directories("../knn_prediction/"+noise);
//            }
//            imwrite("../knn_prediction/"+noise+"/inference_tile_"+to_string(test_images_index[i])+".png", testInputOutput.convert_to_color(predicted_label_maps[i]));
//        }
//    }

    return 0;
}