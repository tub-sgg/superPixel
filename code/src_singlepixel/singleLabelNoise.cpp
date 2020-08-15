
#include "singleLabelNoise.h"

singleLabelNoise::singleLabelNoise(PixelSelectionType selectionType, NewLabelType newLabelType, int number_of_classes, float noiseLevel) {

    this->selectionType = selectionType;
    this->newLabelType = newLabelType;
    this->number_of_classes = number_of_classes;
    this->noiseLevel = noiseLevel;
}



Mat singleLabelNoise::addLabelNoise(const Mat &img, const Mat &labelMap, const MatrixXd &confusionMatrix) {
    Size imgsize = img.size();
    Mat newLabelMap = labelMap.clone();
    if(newLabelType == randomRelabel){

        for(auto &pt:selected_samples){
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev());
            std::uniform_int_distribution<int> class_distribution(0, number_of_classes-1);
            int newlabel;
            do{
                newlabel = class_distribution(generator);
            }while(newlabel == (int)labelMap.at<uchar>(pt.y,pt.x));

            newLabelMap.at<uchar>(pt.y,pt.x) = (uchar)newlabel;

        }
        return newLabelMap;
    }else if(newLabelType == CMrelabel){ // relabel the pixel based on the confusion matrix from KNN classifier

        vector<map<int,float> > mapping;
        for(int i=0;i<number_of_classes;i++){
            map<int,float> single_mapping;
            float sum=0.0;
            for(int j=0;j<number_of_classes;j++){
                if(j != i){
                    sum += confusionMatrix(i,j);
                    single_mapping.insert(make_pair(j,sum));
                }
            }

            map<int, float>::iterator it;

            for(it = single_mapping.begin(); it!=single_mapping.end(); it++){
                it->second /= sum;
            }
            mapping.push_back(single_mapping);
        }

        for(auto &pt:selected_samples){
            int trueLabel = (int)labelMap.at<uchar>(pt.y,pt.x);

            std::random_device rand_dev;
            std::mt19937 generator(rand_dev());
            std::uniform_real_distribution<float> prob_distribution(0.0, 1.0);
            int newlabel=0;
            float probability = prob_distribution(generator);
            map<int, float>::iterator it;

            for(it = mapping.at(trueLabel).begin(); it!=mapping.at(trueLabel).end(); it++){
                if(probability<it->second) {
                    newlabel = it->first;
                    break;
                }
            }
            newLabelMap.at<uchar>(pt.y,pt.x) = (uchar)newlabel;

        }

        return newLabelMap;
    }else if(newLabelType == localNeighbourInReference){
        for(auto &pt:selected_samples){

            int newlabel;
            int patch_size = 5;
            int band_size = (patch_size-1)/2;
            vector<int> neighbourLabels;
            for(int j=0;j<patch_size;j++){
                for(int z=0;z<patch_size;z++){
                    int x = pt.x+z-band_size;
                    int y = pt.y+j-band_size;
                    if (x < 0) x = 0;
                    if (x >= imgsize.width) x = imgsize.width - 1;
                    if (y < 0) y = 0;
                    if (y >= imgsize.height) y = imgsize.height - 1;
                    neighbourLabels.push_back((int)labelMap.at<uchar>(y,x));
                }
            }

            vector<int> histogram = vector<int>(number_of_classes,0);
            for(auto const & neighbourlabel:neighbourLabels){
                histogram[neighbourlabel]+=1;
            }


            int maxElementIndex = max_element(histogram.begin(),histogram.end()) - histogram.begin();
            histogram.at(maxElementIndex) = 0;
            maxElementIndex = max_element(histogram.begin(),histogram.end()) - histogram.begin();
            newLabelMap.at<uchar>(pt.y,pt.x) = (uchar)maxElementIndex;

        }
        return newLabelMap;
    }else if(newLabelType == globalNeighbourInReference){ // relabel the pixel based on global neighbors in reference map
        MatrixXd transitionMatrix = MatrixXd::Zero(number_of_classes, number_of_classes);
        for(int height=0;height<imgsize.height;height++){
            for(int width=0;width<imgsize.width;width++){
                vector<Point> neighbors={Point(width-1, height), Point(width, height-1), Point(width+1, height), Point(width, height+1)};
                for(auto & neighbor:neighbors){
                    if (neighbor.x < 0) break;
                    if (neighbor.x >= imgsize.width) break;
                    if (neighbor.y < 0) break;
                    if (neighbor.y >= imgsize.height) break;
                    transitionMatrix((int)labelMap.at<uchar>(height,width), (int)labelMap.at<uchar>(neighbor)) +=1;
                }
            }
        }
        vector<map<int,float> > mapping;
        for(int i=0;i<number_of_classes;i++){
            map<int,float> single_mapping;
            float sum=0.0;
            for(int j=0;j<number_of_classes;j++){
                if(j != i){
                    sum += transitionMatrix(i,j);
                    single_mapping.insert(make_pair(j,sum));
                }
            }

            map<int, float>::iterator it;

            for(it = single_mapping.begin(); it!=single_mapping.end(); it++){
                it->second /= sum;
            }

            mapping.push_back(single_mapping);
        }

        for(auto &pt:selected_samples){
            int trueLabel = (int)labelMap.at<uchar>(pt.y,pt.x);

            std::random_device rand_dev;
            std::mt19937 generator(rand_dev());
            std::uniform_real_distribution<float> prob_distribution(0.0, 1.0);
            int newlabel=0;
            float probability = prob_distribution(generator);
            map<int, float>::iterator it;

            for(it = mapping.at(trueLabel).begin(); it!=mapping.at(trueLabel).end(); it++){
                if(probability<it->second) {
                    newlabel = it->first;
                    break;
                }
            }
            newLabelMap.at<uchar>(pt.y,pt.x) = (uchar)newlabel;

        }
        return newLabelMap;

    }else if(newLabelType == neighbourInFeatureSpace){
    // relabel the pixel based on the neighbors in feature space
        int patch_size = 3;
        int band_size = (patch_size-1)/2;
        Mat train_data;
        Mat train_labels;
        for(int a=0;a<imgsize.height;a++){
            for(int b=0;b<imgsize.width;b++){
                if(find(selected_samples.begin(), selected_samples.end(), Point(b,a)) != selected_samples.end()) break;

                Mat feature_single_sample;
                for(int j=0;j<patch_size;j++){
                    for(int z=0;z<patch_size;z++){
                        int x = b+z-band_size;
                        int y = a+j-band_size;
                        if (x < 0) x = 0;
                        if (x >= imgsize.width) x = imgsize.width - 1;
                        if (y < 0) y = 0;
                        if (y >= imgsize.height) y = imgsize.height - 1;
                        feature_single_sample.push_back(img.at<Vec3b>(y,x)[0]);
                        feature_single_sample.push_back(img.at<Vec3b>(y,x)[1]);
                        feature_single_sample.push_back(img.at<Vec3b>(y,x)[2]);
                    }
                }

                feature_single_sample.convertTo(feature_single_sample, CV_32F);
                train_data.push_back(feature_single_sample.reshape(1,1));
                train_labels.push_back((int) labelMap.at<uchar>(b, a));

            }
        }
        Ptr<ml::KNearest> knn;
        knn = ml::KNearest::create();
        knn->train(train_data, ml::ROW_SAMPLE, train_labels);

        for(const auto & pt:selected_samples){
            Mat feature_single_sample;
            for(int j=0;j<patch_size;j++){
                for(int z=0;z<patch_size;z++){
                    int x = pt.x+z-band_size;
                    int y = pt.y+j-band_size;
                    if (x < 0) x = 0;
                    if (x >= imgsize.width) x = imgsize.width - 1;
                    if (y < 0) y = 0;
                    if (y >= imgsize.height) y = imgsize.height - 1;
                    feature_single_sample.push_back(img.at<Vec3b>(y,x)[0]);
                    feature_single_sample.push_back(img.at<Vec3b>(y,x)[1]);
                    feature_single_sample.push_back(img.at<Vec3b>(y,x)[2]);
                }
            }
            feature_single_sample.convertTo(feature_single_sample, CV_32F);
            Mat test_data;
            test_data.push_back(feature_single_sample.reshape(1,1));
            Mat predicted_label, neighbours;
            int K = 51;
            knn->findNearest(test_data, K, predicted_label, neighbours);

            int true_label = (int)labelMap.at<uchar>(pt.y, pt.x);
            int predict_label = (int)predicted_label.at<float>(0);

            if(predict_label==true_label){
                vector<int> histogram = vector<int>(number_of_classes,0);
                for(int i=0;i<K;i++){
                    histogram[(int)neighbours.at<float>(0,i)]+=1;
                }
                histogram[true_label] = 0;
                int maxElementIndex = std::max_element(histogram.begin(),histogram.end()) - histogram.begin();
                newLabelMap.at<uchar>(pt.y,pt.x) = (uchar)maxElementIndex;
            }else{
                newLabelMap.at<uchar>(pt.y,pt.x) = (uchar)predict_label;
            }
        }

        return newLabelMap;
    }else cerr<<"please input correct for relabelling"<<endl;

}


void singleLabelNoise::sampleSelection(const Mat &uncertainty, const Mat  &labelMap) {
    Size imgsize = labelMap.size();
    int relabelSamples =(int)(noiseLevel*(float)imgsize.height*(float)imgsize.width);

    if(selectionType==randomSelection){ // randomly select 10% amount of pixels
        vector<Point> all_samples;
        for(int height=0;height<imgsize.height;height++){
            for(int width=0;width<imgsize.width;width++){
                all_samples.push_back(Point(width,height));
            }
        }
        random_device rand_dev;
        mt19937 generator(rand_dev());
        shuffle(all_samples.begin(), all_samples.end(), generator);


        int count=0;
        for(auto &sample:all_samples){
            if(count<relabelSamples) {
                selected_samples.push_back(sample);
                count++;
            }
            else break;
        }

    }else if(selectionType==uncertaintyArea){ // select pixels that have high uncertainty based on the KNN classificaion

        Mat blurredUncertaintyMap;
        GaussianBlur(uncertainty,blurredUncertaintyMap, Size(5,5), 0);

        int count = 0;
        do{
//            double minVal;
//            double maxVal;
//            Point minLoc;
            Point maxLoc;
            minMaxLoc(blurredUncertaintyMap, nullptr, nullptr, nullptr, &maxLoc );
            blurredUncertaintyMap.at<uchar>(maxLoc) = 0;
            selected_samples.push_back(maxLoc);
            count ++;
        }while(count<relabelSamples);


    }else if(selectionType==objectBorder){ // select pixel from object boundaries
        Mat labelMap_gray;
        cvtColor(labelMap, labelMap_gray, CV_BGR2GRAY);
        Mat binaryMask;
        Canny(labelMap_gray, binaryMask, 20, 50);

        Mat blurredbinaryMask, binarizedBlurredMask;
        GaussianBlur(binaryMask,blurredbinaryMask, Size(5,5), 0);
        threshold(blurredbinaryMask,binarizedBlurredMask, 0, 255, THRESH_BINARY);

        vector<Point> all_samples;
        for(int height=0;height<imgsize.height;height++){
            for(int width=0;width<imgsize.width;width++){
                if((int)binarizedBlurredMask.at<uchar>(height,width)>128) all_samples.push_back(Point(width,height));
            }
        }
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev());
        shuffle(all_samples.begin(), all_samples.end(), generator);
        int count=0;
        for(auto &sample:all_samples){
            if(count<relabelSamples) {
                selected_samples.push_back(sample);
                count++;
            }
            else break;
        }
    }else cerr<<"please input correct way to selection samples to relabel"<<endl;

}
