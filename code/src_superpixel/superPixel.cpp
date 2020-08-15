//
// Created by alxlee on 05.07.20.
//

#include "superPixel.h"

namespace SuperPixel {
    cv::Mat superPixel::convert_from_color(const cv::Mat &src) {
        cv::Mat label_map = cv::Mat::zeros(src.size(), src.type());
        for (int i = 0; i < label_map.size().height; i++) {
            for (int j = 0; j < label_map.size().width; j++) {
                //white
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                //blue
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 0, 0))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(1, 1, 1);
                //light blue
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 0))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(2, 2, 2);
                //green
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 255, 0))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(3, 3, 3);
                //yellow
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 255, 255))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(4, 4, 4);
                //red
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 255))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(5, 5, 5);
            }
        }
        return label_map;
    }

    cv::Mat superPixel::convert_to_color(const cv::Mat &src) {
        cv::Mat label_map_color_coded = cv::Mat::zeros(src.size(), src.type());
        for (int i = 0; i < label_map_color_coded.size().height; i++) {
            for (int j = 0; j < label_map_color_coded.size().width; j++) {
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0))
                    label_map_color_coded.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(1, 1, 1))
                    label_map_color_coded.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(2, 2, 2))
                    label_map_color_coded.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 0);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(3, 3, 3))
                    label_map_color_coded.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(4, 4, 4))
                    label_map_color_coded.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 255);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(5, 5, 5))
                    label_map_color_coded.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
        }
        return label_map_color_coded;

    }

    void superPixel::getLocalNeighbourMap() {
        std::cout<<"start getLocalNeighbourMap***************"<<std::endl;
        clock_t startTime,endTime;
        startTime = clock();
        localneighborMap.resize(MaximumSuperPixelIndex,0);
        //extend label map
        //extend 2 pixels larger
        cv::Mat ext_labelMap;
        ext_labelMap.create(image.rows + 2, image.cols + 2, image.type());
        ext_labelMap.setTo(cv::Vec3b(255, 255, 255));//set a value not same as labels
        image.copyTo(ext_labelMap(cv::Rect(1, 1, image.cols, image.rows)));

        //for each super pixel, calculate the neighbors

//#pragma omp parallel for
        for(auto superpixel=0;superpixel<MaximumSuperPixelIndex;superpixel++){
            LABEL super_label=preprocessSuperPixel(allsuperPixelCoordinates[superpixel]);
            std::vector<int> vec;
            vec.resize(ls.size(),0);
            cv::Mat tmp=ext_labelMap.clone();
            for(const auto& pt:allsuperPixelCoordinates[superpixel])
                tmp.at<cv::Vec3b>(cv::Point(pt.x+1,pt.y+1))=cv::Vec3b(255,255,255);
            for(const auto& pt:allsuperPixelCoordinates[superpixel]){
                cv::Point top=cv::Point(pt.x+1,pt.y);
                cv::Point left=cv::Point(pt.x,pt.y+1);
                cv::Point right=cv::Point(pt.x+2,pt.y+1);
                cv::Point bot=cv::Point(pt.x+1,pt.y+2);

                LABEL l_top=tmp.at<cv::Vec3b>(top)[0];
                if(l_top!=255) vec[l_top]+=1;
                LABEL l_left=tmp.at<cv::Vec3b>(left)[0];
                if(l_left!=255) vec[l_left]+=1;
                LABEL l_right=tmp.at<cv::Vec3b>(right)[0];
                if(l_right!=255) vec[l_right]+=1;
                LABEL l_bot=tmp.at<cv::Vec3b>(bot)[0];
                if(l_bot!=255) vec[l_bot]+=1;
            }
            //find the max
            auto maxEle=std::max_element(vec.begin(),vec.end());
            LABEL predict=std::distance(vec.begin(),maxEle);
            int loc=superpixel;
            if(*maxEle==0)
                localneighborMap[loc]=super_label;
            else
                localneighborMap[loc]=predict;
        }
        endTime = clock();
        std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        std::cout<<"end getLocalNeighbourMap***************"<<std::endl;

    }
//return the label of majority pixel in superpixls
    LABEL superPixel::preprocessSuperPixel(const std::vector<cv::Point> &superpixel) {
        const int labelNumber = ls.size();
        std::vector<int> st(labelNumber, 0);
        for (const auto &pt:superpixel) {
            auto index = (LABEL) image.at<cv::Vec3b>(pt)[0];
            st[index] += 1;
        }
        auto it = std::max_element(st.begin(), st.end());
        LABEL label = std::distance(st.begin(),it);

        return label;
    }

    void superPixel::extract_Contour() {
        //convert 3 channel to 1 channel
        std::vector<cv::Mat> SrcMatpart(image.channels());

        cv::Mat img;
        cv::cvtColor(convert_to_color(image),img,cv::COLOR_BGR2GRAY);

        cv::Canny(img,img,20,50);

        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE, cv::Point());
    }

    void superPixel::calculateNeighborNumber() {
        NeighborMatrix.create(cv::Size(ls.size(), ls.size()));
        NeighborMatrix.setTo(0);
        //extend 2 pixels larger
        cv::Mat ext_labelMap;
        ext_labelMap.create(image.rows + 2, image.cols + 2, image.type());
        ext_labelMap.setTo(cv::Vec3b(255, 255, 255));//set a value not same as labels
        image.copyTo(ext_labelMap(cv::Rect(1, 1, image.cols, image.rows)));
        //only need lower part of the matrix
        for (int r = 1; r < ext_labelMap.rows - 1; r++)
            for (int c = 1; c < ext_labelMap.cols - 1; c++) {
                int top = (int) ext_labelMap.at<cv::Vec3b>(r - 1, c)[0];
                int left = (int) ext_labelMap.at<cv::Vec3b>(r, c - 1)[0];
                int right = (int) ext_labelMap.at<cv::Vec3b>(r, c + 1)[0];
                int bot = (int) ext_labelMap.at<cv::Vec3b>(r + 1, c)[0];

                LABEL l = (LABEL) ext_labelMap.at<cv::Vec3b>(r, c)[0];


                if (top != l and top != 255  ) { NeighborMatrix(l, top)+=1; }
                if (left != l and left != 255 ) { NeighborMatrix(l, left)+=1; }
                if (right != l and right != 255 ) { NeighborMatrix(l, right)+=1; }
                if (bot != l and bot != 255 ) { NeighborMatrix(l, bot)+=1; }
            }
    }

    void superPixel::getSuperPixelImage(cv::Mat &mask, int &labelNumbers) {
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(image,
                                                                                        cv::ximgproc::SLIC,
                                                                                        superPixelSize);
        slic->iterate();
        slic->enforceLabelConnectivity();
        slic->getLabelContourMask(mask);
        slic->getLabels(labelMap);
        labelNumbers = slic->getNumberOfSuperpixels();
    }

    void superPixel::getSuperPixelCoordinates() {
        std::cout<<"*************************START GET SUPERPIXEL COOR SELECT***********"<<std::endl;
        clock_t startTime,endTime;
        startTime = clock();
        cv::Mat mask;
        int labelNumber;
        getSuperPixelImage( mask, labelNumber);
        allsuperPixelCoordinates.resize(labelNumber);
        for (int r = 0; r < labelMap.size().height; r++)
            for (int c = 0; c < labelMap.size().width; c++){
                int label=(int)labelMap.at<int>(r,c);
                allsuperPixelCoordinates[label].push_back(cv::Point(c,r));
            }

        MaximumSuperPixelIndex=labelNumber;
        endTime = clock();
        std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        std::cout<<"*************************END GET SUPERPIXEL COOR SELECT***************"<<std::endl;
    }

    void superPixel::getBoundaryMap() {
        /***************start calculate contour map**************************/
        clock_t startTime,endTime;
        startTime = clock();
        this->contourMap=cv::Mat::zeros(image.size(),image.type());
        for(const auto&contour:contours){
            for(const auto& pt:contour){
                this->contourMap.at<uchar>(pt)=1;
            }
        }
        endTime = clock();
        std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        /***************end calculate contour map**************************/
    }

    void superPixel::calculateSuperpixelRelationshipMap() {
        std::cout<<"*************************START SuperpixelRelationshipMap***********"<<std::endl;
        clock_t startTime,endTime;
        startTime = clock();
        superpixelRelationMap.resize(MaximumSuperPixelIndex);
        std::vector<std::set<int>> vec;
        vec.resize(MaximumSuperPixelIndex);
        cv::Mat ext_labelMap;
        cv::copyMakeBorder(labelMap,ext_labelMap,1,1,1,1,cv::BORDER_REPLICATE);
        int number=ext_labelMap.size().height*ext_labelMap.size().width;
#pragma omp parallel for
        for(int i=0;i<number;i++)
            {
            int r = i / ext_labelMap.size().width;
            int c = i % ext_labelMap.size().width;
                if (r != 0 and r != (ext_labelMap.size().height-1) and c != 0 and c != (ext_labelMap.size().width-1)) {
                    cv::Point top = cv::Point(c, r - 1);
                    cv::Point left = cv::Point(c - 1, r);
                    cv::Point right = cv::Point(c + 1, r);
                    cv::Point bot = cv::Point(c, r + 1);
                    int t = ext_labelMap.at<int>(top);
                    int l = ext_labelMap.at<int>(left);
                    int rg = ext_labelMap.at<int>(right);
                    int b = ext_labelMap.at<int>(bot);
                    int loc = ext_labelMap.at<int>(r, c);
                    if (loc != t) vec[loc].insert(t);
                    if (loc != l) vec[loc].insert(l);
                    if (loc != rg) vec[loc].insert(rg);
                    if (loc != b) vec[loc].insert(b);

                }
            }
        for(auto index=0;index<MaximumSuperPixelIndex;index++){
            auto it=vec[index].begin();
            for(;it!=vec[index].end();it++){
                superpixelRelationMap[index].push_back(*it);
            }
        }

        endTime = clock();
        std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        /***************end SuperpixelRelationshipMap**************************/
        std::cout<<"*************************END SuperpixelRelationshipMap***********"<<std::endl;
    }

    void superPixel::sampleSelection(int N) {
        CV_Assert(N<MaximumSuperPixelIndex);
        selectedSuperPixelIndex.resize(N);
        switch (selectionType) {
            case randomSelection: {
                std::cout<<"*************************START RANDOM SELECT***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();

                std::vector<int> vec(MaximumSuperPixelIndex,0);
                std::generate(vec.begin(), vec.end(), [n = 0]() mutable {return n++;});

                std::random_device rd;
                std::mt19937 gen(rd());
                std::shuffle(vec.begin(), vec.end(), gen);
                auto it = vec.begin();
                std::copy(it,it+N,selectedSuperPixelIndex.begin());

                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END RANDOM SELECT***************"<<std::endl;

                break;
            }

            case uncertaintyArea: {
                std::cout<<"*************************START UNCERTAINTY SELECT***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();

                //smooth uncertainty map
                cv::Mat_<cv::Vec3b> gaussianBlurUncertainty;
                cv::Size kernel=cv::Size(5,5);
                cv::GaussianBlur(uncertainty,gaussianBlurUncertainty,kernel,0);

                std::multimap<int,int,std::greater<int> > uncertaintyIndexMap;
                //average threshold
                for (auto index=0;index<MaximumSuperPixelIndex;index++) {
                    int value = 0;
                    for (const auto &pt:allsuperPixelCoordinates[index]) {
                        value += (int) gaussianBlurUncertainty(pt)[0] ;
                    }
                    uncertaintyIndexMap.insert(std::make_pair(value,index));
                }
                std::vector<int> allSuperPixelIndex;
                for(const auto& uncer:uncertaintyIndexMap){
                    allSuperPixelIndex.push_back(uncer.second);
                }

                auto it=allSuperPixelIndex.begin();
                std::copy(it,it+N,selectedSuperPixelIndex.begin());

                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END UNCERTAINTY SELECT***************"<<std::endl;
                break;
            }

            case objectBorder: {
                std::cout<<"*************************START BOUNDARY SELECT***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();
                selectedSuperPixelIndex.resize(N);
                std::vector<int> allsuperPixelIntersectContour;

                std::vector<int> vec(MaximumSuperPixelIndex,0);
                std::generate(vec.begin(), vec.end(), [n = 0]() mutable {return n++;});

                for(const auto& superpixel:vec){
                    for(const auto& pt:allsuperPixelCoordinates[superpixel]){
                        int value=(int)contourMap.at<uchar>(pt);
                        if(value==1){
                            allsuperPixelIntersectContour.push_back(superpixel);
                            break;
                        }
                    }
                }
                int superpixelonboundarysize=allsuperPixelIntersectContour.size();
                if(N<superpixelonboundarysize){
                    std::random_device RD;
                    std::mt19937 GEN(RD());
                    std::shuffle(allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end(),GEN);
                    auto it=allsuperPixelIntersectContour.begin();
                    std::copy(it,it+N,selectedSuperPixelIndex.begin());
                }else if(N==superpixelonboundarysize){
                    auto it=allsuperPixelIntersectContour.begin();
                    std::copy(it,it+N,selectedSuperPixelIndex.begin());
                }else{
                    //find the super pixel close to the selected super pixels
                    std::cout<<"the number of superpixel located on the boundary is not enough"<<std::endl;
                    std::cout<<"*************************START EXTRAL PIXEL SELECT***************"<<std::endl;
                    clock_t starttime1=clock();
                    calculateSuperpixelRelationshipMap();
                    cv::Mat_<int> updateSelectedMap=cv::Mat_<int>::zeros(image.size());

                    std::vector<int> newindexvector(allsuperPixelIntersectContour);
                    std::vector<int> oldvector;
                    oldvector.insert(oldvector.begin(),allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end());
                    do{
                        std::vector<int> newvector;
                        int sizenewindex=newindexvector.size();
                        auto newindexvector1=newindexvector;
                        auto it=newindexvector.begin();
                        for(;it!=newindexvector.end();it++) {

                            newindexvector1.insert(newindexvector1.end(),superpixelRelationMap[*it].begin(),
                                                   superpixelRelationMap[*it].end());

                        }

                        std::sort(newindexvector1.begin(),newindexvector1.end());
                        newindexvector1.erase(std::unique(newindexvector1.begin(),newindexvector1.end()),newindexvector1.end());
                        //insert to old
                        oldvector.insert(oldvector.begin(),newindexvector1.begin(),newindexvector1.end());
                        std::sort(oldvector.begin(),oldvector.end());
                        oldvector.erase(std::unique(oldvector.begin(),oldvector.end()),oldvector.end());
                        int s=oldvector.size();
                        std::cout<<"oldvector size="<<s<<std::endl;
                        superpixelonboundarysize=N-oldvector.size();
                        if (superpixelonboundarysize == 0){
                            std::copy(allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end(),
                                      selectedSuperPixelIndex.begin());
                            break;
                        }
                        if(superpixelonboundarysize<0){
                            selectedSuperPixelIndex.insert(selectedSuperPixelIndex.begin(),
                                                           oldvector.begin(),oldvector.end());
                            std::vector<int> rest;
                            std::set_difference(oldvector.begin(),oldvector.end(),
                                                allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end(),std::inserter(rest,rest.begin()));

                            std::random_device RD;
                            std::mt19937 GEN(RD());
                            std::shuffle(rest.begin(),rest.end(),GEN);
                            auto s1=rest.size();
                            auto it1=rest.begin();
                            auto len=N-allsuperPixelIntersectContour.size();
                            selectedSuperPixelIndex.insert(selectedSuperPixelIndex.end(),it1,it1+len);
                            break;
                        }

                        newindexvector.clear();
                        newindexvector=newindexvector1;
                        newindexvector1.clear();

                    }while(superpixelonboundarysize>0);
                    endTime = clock();
                    std::cout << "Totle Time : " <<(double)(endTime - starttime1) / CLOCKS_PER_SEC << "s" << std::endl;
                    std::cout<<"*************************END EXTRAL PIXEL SELECT***************"<<std::endl;
                }
                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END BOUNDARY SELECT***************"<<std::endl;
                break;
            }
            default: {
                std::cerr << "something wrong with selection!!!" << std::endl;
            }
        }
    }

    void superPixel::addLabelNoise() {
        switch (addLabelNoiseType) {
            case randomRelabel: {
                std::cout<<"*************************START RANDOM RELABEL***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();
                CV_Assert(image.channels() == 3);

                addedLabelNoiseImage.create(image.size(), image.type());
                this->image.copyTo(addedLabelNoiseImage);
                std::cout<<"random_superpixel size="<<this->addedLabelNoiseImage.size()<<std::endl;
                //copy label list
                std::vector<LABEL> l;
                for (auto it:ls)
                    l.push_back(it);

                for (auto const &sp:selectedSuperPixelIndex) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::shuffle(l.begin(), l.end(), gen);
                    LABEL label = l[0];
                    for (auto const &pt:allsuperPixelCoordinates[sp]) {
                        addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(label, label, label);
                    }
                }
                std::cout << l[0] << std::endl;

                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END RANDOM RELABEL***************"<<std::endl;

                break;
            }

            case CMrelabel: {//confusionMatrix
                std::cout<<"*************************START CONFUSION RELABEL***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();

                CV_Assert(image.channels() == 3);
                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);

                cv::Mat_<float> confusionmatrix;
                confusionmatrix.create(confusionMatrix.size());
                confusionMatrix.copyTo(confusionmatrix);
                // the true label can be most set by the label based on the row
                // set the diag value 0
                for(int r = 0; r < confusionmatrix.size().height; r++)
                    for (int c = 0; c < confusionmatrix.size().width; c++)
                    {
                        if(r==c)
                            confusionmatrix(r,c)=0.f;
                    }
      ;
                //find the most confused label

                std::vector<int> confusedMap;//the first is the true label the sencond is the confused label
                for (int r = 0; r < confusionmatrix.size().height; r++) {
                    cv::Point  maxLoc;
                    cv::Mat row=confusionmatrix.row(r);
                    double maxv;
                    cv::minMaxLoc(row, nullptr, &maxv, nullptr,&maxLoc);
                    confusedMap.push_back(maxLoc.x);
                }

                if (confusedMap.empty())
                    return;
                else {
                    for (const auto &sp:selectedSuperPixelIndex) {
                        LABEL label=preprocessSuperPixel(allsuperPixelCoordinates[sp]);
                        LABEL relabel=confusedMap[label];
                        for (const auto &pt:allsuperPixelCoordinates[sp]) {
                            addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b( relabel,relabel,relabel);
                        }

                    }
                }

                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END CONFUSION RELABEL***************"<<std::endl;

                break;
            }

            case GlobalNeighbour: {//how often(GLOBAL)
                std::cout<<"*************************START Global NEIGHBOUR RELABEL***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();

                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);

                std::cout<<"NeighborMatrix="<<NeighborMatrix<<std::endl;
                std::vector<int> MostMap;
                MostMap.resize(ls.size(),0);
                for (int r = 0; r < NeighborMatrix.size().height; r++) {
                    cv::Point  maxLoc;
                    cv::Mat row=NeighborMatrix.row(r);
                    double maxv;
                    cv::minMaxLoc(row, nullptr, &maxv, nullptr,&maxLoc);
                    if(maxv>0.f)
                    MostMap[r]=maxLoc.x;
                    else{
                        MostMap[r]=r;
                    }
                }

                for (const auto &superpixel:selectedSuperPixelIndex) {
                    LABEL l = preprocessSuperPixel(allsuperPixelCoordinates[superpixel]);
                        LABEL relabel=MostMap[l];
                        for (const auto &pt:allsuperPixelCoordinates[superpixel]) {
                            addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(relabel,relabel,relabel);
                        }
                }
                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END Global NEIGHBOUR RELABEL***************"<<std::endl;

                break;
            }
            case localNeighbour:{//local most
                std::cout<<"*************************START LOCAL NEIGHBOUR RELABEL***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();
                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);

                for(const auto& superpixel:selectedSuperPixelIndex) {
                    int loc=superpixel;
                    LABEL relabel=localneighborMap[loc];
                    for (const auto &pt:allsuperPixelCoordinates[superpixel]) {
                        addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(relabel,relabel,relabel);
                    }
                }
                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END LOCAL NEIGHBOUR RELABEL***************"<<std::endl;

                break;
            }
            case NearestNeighbour: {//nearest neighbor==  in Feature Space using knn
                std::cout<<"*************************START NEAREST NEIGHBOUR RELABEL***************"<<std::endl;
                clock_t startTime,endTime,endtime1,endtime2,endtime3,endtime4;
                startTime = clock();

                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);

                int patch_size = 3;
                int band_size = (patch_size-1)/2;
                cv::Mat trainData;
                cv::Mat trainLabels;
                cv::Size size=image.size();

                //crate selected map
                cv::Mat_<int> selectedMap=cv::Mat_<int>::zeros(featureImg.size());
                for(const auto superPixel:selectedSuperPixelIndex)
                    for(const auto& pt:allsuperPixelCoordinates[superPixel]){
                        selectedMap(pt)=1;
                    }

                //find the smallest superpixel size
                int smallest=allsuperPixelCoordinates[0].size();
                for(int index=1;index<MaximumSuperPixelIndex;index++){
                    if(allsuperPixelCoordinates[index].size()<smallest)
                        smallest=allsuperPixelCoordinates[index].size();
                }
                std::cout<<"smallest="<<smallest<<std::endl;

                for(int index=0;index<MaximumSuperPixelIndex;index++){
                    int x=index%size.width;
                    int y=index/size.width;
                    cv::Point p=cv::Point(x,y);
                    if(selectedMap(p)==0){
                        cv::Mat features_pixel;
                        std::vector<cv::Point> superP(allsuperPixelCoordinates[index]);

                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::shuffle(superP.begin(), superP.end(), gen);

                        std::vector<cv::Point> superP1;
                        superP1.resize(smallest);

                        auto it = superP.begin();
                        std::copy(it,it+smallest,superP1.begin());

                        for(const auto&pt:superP1){
                            cv::Point p1=cv::Point(pt);
                            features_pixel.push_back(featureImg.at<cv::Vec3b>(p1)[0]);
                            features_pixel.push_back(featureImg.at<cv::Vec3b>(p1)[1]);
                            features_pixel.push_back(featureImg.at<cv::Vec3b>(p1)[2]);
                        }
                        features_pixel.convertTo(features_pixel, CV_32F);
                        trainData.push_back(features_pixel.reshape(1,1));
                        LABEL l=preprocessSuperPixel(allsuperPixelCoordinates[index]);
                        trainLabels.push_back(l);
                    }

                }
                std::cout<<"*************************START KNN TRAIN ***************"<<std::endl;
                endtime1=clock();
                cv::Ptr<cv::ml::KNearest> knn=cv::ml::KNearest::create();
                knn->train(trainData,cv::ml::ROW_SAMPLE,trainLabels);
                endtime2=clock();
                std::cout << "Totle Time : " <<(double)(endtime2 - endtime1) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END KNN TRAIN *****************"<<std::endl;

                std::cout<<"*************************START FIND NEAREST  ***************"<<std::endl;
                endtime3=clock();

                for(const auto& index:selectedSuperPixelIndex){
                    int x=index%size.width;
                    int y=index/size.width;
                    cv::Point p=cv::Point(x,y);

                    cv::Mat features_pixel;
                    std::vector<cv::Point> superP(allsuperPixelCoordinates[index]);

                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::shuffle(superP.begin(), superP.end(), gen);

                    std::vector<cv::Point> superP1;
                    superP1.resize(smallest);

                    auto it = superP.begin();
                    std::copy(it,it+smallest,superP1.begin());

                    for(const auto&pt:superP1){
                        cv::Point p1=cv::Point(pt);
                        features_pixel.push_back(featureImg.at<cv::Vec3b>(p1)[0]);
                        features_pixel.push_back(featureImg.at<cv::Vec3b>(p1)[1]);
                        features_pixel.push_back(featureImg.at<cv::Vec3b>(p1)[2]);
                    }
                    features_pixel.convertTo(features_pixel, CV_32F);
                    cv::Mat testData;
                    testData.push_back(features_pixel.reshape(1, 1));
                    cv::Mat predictedLabel;
                    cv::Mat neighbours;
                    int K = 51;
                    knn->findNearest(testData, K, predictedLabel,neighbours);
                    LABEL truelabel=preprocessSuperPixel(allsuperPixelCoordinates[index]);
                    LABEL predic=(LABEL)predictedLabel.at<float>(0);
                    if(truelabel!=predic){
                        for(const auto&pt:allsuperPixelCoordinates[index]){
                            addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b(predic,predic,predic);
                        }
                    }else{
                        //get the second nearest neighbor
                        std::vector<LABEL> his(ls.size(), 0);
                        for (auto j = 0; j < neighbours.size().width; j++) {
                            LABEL l = (LABEL) neighbours.at<float>(0, j);
                            his[l] += 1;
                        }
                        auto itor = std::max_element(his.begin(), his.end());
                        //if there is only one neighbor
                        LABEL lab=*itor;
                        if(lab==K){
                            for(const auto&pt:allsuperPixelCoordinates[index]){
                                addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b(truelabel,truelabel,truelabel);
                            }
                            //printf("only one label found=%d \n",lab);
                        }else{
                            //find the nearest
                            int tp1=*itor;//the number of first largest
                            LABEL l1=std::distance(his.begin(),itor);
                            *itor=-1;
                            auto it2=std::max_element(his.begin(),his.end());
                            auto tp2=*it2;
                            LABEL l2=std::distance(his.begin(),it2);
                            if(tp1==tp2){
                                if(l1==truelabel){
                                    for(const auto&pt:allsuperPixelCoordinates[index]){
                                        addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b(l2,l2,l2);
                                    }
                                    //printf("two identical maximum number found ,the second =%d \n",l2);
                                }else{
                                    for(const auto&pt:allsuperPixelCoordinates[index]){
                                        addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b(l1,l1,l1);
                                    }
                                    //printf("two identical maximum number found, the first =%d \n",l1);
                                }
                            }else{
                                for(const auto&pt:allsuperPixelCoordinates[index]){
                                    addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b(l2,l2,l2);
                                }
                                //printf("the second nearest label found =%d \n",l2);
                            }
                        }
                    }



                }
                endtime4=clock();
                std::cout << "Totle Time : " <<(double)(endtime4 - endtime3) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END FIND NEAREST  ***************"<<std::endl;
                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END NEAREST NEIGHBOUR RELABEL***************"<<std::endl;
                break;
            }
            default: {
                std::cerr << "something wrong with relabel selection" << std::endl;
            }
        }
    }
    void superPixel::saveColorImage( const std::string& filename) {
        cv::Mat img=convert_to_color(this->addedLabelNoiseImage);

        cv::imwrite(filename+".png",img);
    }

    void superPixel::superPixelNoise(SuperPixel::options option,std::string& outputFilenameIndex,int N) {
        switch(option){
            case RandomSelectRandomRelabel:{
                //**********Select method:random selection********************
                //**********Relabel method:random selection*******************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::randomRelabel);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectRandomRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case RandomSelectGlobalNeighbourRelabel:{
                //**********Select method:random selection********************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::GlobalNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectGlobalNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case RandomSelectConfusionRelabel:{
                //**********Select method:random selection********************
                //**********Relabel method:confusion selection****************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::CMrelabel);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectConfusionRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }

            case RandomSelectNearestNeighbourRelabel:{
                //**********Select method:random selection********************
                //**********Relabel method:confusion selection****************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::NearestNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectNearestNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case RandomSelectLocalNeighbourRelabel:{
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::localNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectLocalNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }

            case UncertaintyAreaSelectRandomRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:Random selection*******************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::randomRelabel);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectRandomRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case UncertaintyAreaSelectConfusionRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:confusion selection****************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::CMrelabel);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectConfusionRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case UncertaintyAreaSelectGlobalNeighbourRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::GlobalNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectGlobalNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case UncertaintyAreaSelectNearestNeighbourRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:neighbourInFeatureSpace selection**
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::NearestNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectNearestNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case UncertaintyAreaSelectLocalNeighbourRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:neighbourInFeatureSpace selection**
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::localNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectLocalNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case ObjectBorderSelectRandomRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:randomRelabel selection************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::randomRelabel);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectRandomRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case ObjectBorderSelectConfusionRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:confusion selection****************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::CMrelabel);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectConfusionRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }

            case ObjectBorderSelectGlobalNeighbourRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::GlobalNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectGlobalNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case ObjectBorderSelectNearestNeighbourRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:neighbourInFeatureSpace selection**
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::NearestNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectNearestNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case ObjectBorderSelectLocalNeighbourRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:neighbourInFeatureSpace selection**
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::localNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectLocalNeighbourRelabel"+outputFilenameIndex);
                //clear the data
                selectedSuperPixelIndex.clear();
                break;
            }
            case RandomSelectionAllRelabel:{
                //select data
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                sampleSelection(N);

                //random relabel
                setAddLabelNoiseType(SuperPixel::NewLabelType::randomRelabel);
                addLabelNoise();
                //save color image

                saveColorImage("RandomSelectRandomRelabel"+outputFilenameIndex);

                //confusion relabel
                setAddLabelNoiseType(SuperPixel::NewLabelType::CMrelabel);
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectConfusionRelabel"+outputFilenameIndex);

                //Global neighor
                setAddLabelNoiseType(SuperPixel::NewLabelType::GlobalNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectGlobalNeighbourRelabel"+outputFilenameIndex);

                //nearest neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::NearestNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectNearestNeighbourRelabel"+outputFilenameIndex);

                //local neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::localNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectLocalNeighbourRelabel"+outputFilenameIndex);
                //clear data
                selectedSuperPixelIndex.clear();

                break;

            }
            case UncertaintySelectAllRelabel:{
                //select data
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                sampleSelection(N);

                //random relabel
                setAddLabelNoiseType(SuperPixel::NewLabelType::randomRelabel);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectRandomRelabel"+outputFilenameIndex);

                //confusion relabel
                setAddLabelNoiseType(SuperPixel::NewLabelType::CMrelabel);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectConfusionRelabel"+outputFilenameIndex);

                //Global neighor
                setAddLabelNoiseType(SuperPixel::NewLabelType::GlobalNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectGlobalNeighbourRelabel"+outputFilenameIndex);

                //nearest neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::NearestNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectNearestNeighbourRelabel"+outputFilenameIndex);

                //local neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::localNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectLocalNeighbourRelabel"+outputFilenameIndex);

                //clear data
                selectedSuperPixelIndex.clear();

                break;
            }
            case ObjectBorderSelectAllRelabel:{
                //select data
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                sampleSelection(N);

                //random relabel
                setAddLabelNoiseType(SuperPixel::NewLabelType::randomRelabel);
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectRandomRelabel"+outputFilenameIndex);

                //confusion relabel
                setAddLabelNoiseType(SuperPixel::NewLabelType::CMrelabel);
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectConfusionRelabel"+outputFilenameIndex);

                //Global neighor
                setAddLabelNoiseType(SuperPixel::NewLabelType::GlobalNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectGlobalNeighbourRelabel"+outputFilenameIndex);

                //nearest neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::NearestNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectNearestNeighbourRelabel"+outputFilenameIndex);

                //local neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::localNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectLocalNeighbourRelabel"+outputFilenameIndex);

                //clear data
                selectedSuperPixelIndex.clear();

                break;
            }

            default:{
                std::cerr<<"error option selection"<<std::endl;
            }

        }

    }


}