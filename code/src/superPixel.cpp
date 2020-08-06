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

    int superPixel::miniColSuperPixel(std::vector<cv::Point> &superPixel) {
        std::vector<int>  vec;
        for(const auto& pt: superPixel){
            int vlaue=pt.x;
            vec.push_back(vlaue);
        }
        auto it=std::min_element(vec.begin(),vec.end());
        return *it;
    }
#if 0
    int superPixel::getPixelNumberinContours() const {
        std::vector<int> vec;
        for(const auto& con:contours)
            vec.push_back(con.size());
        return std::accumulate(vec.begin(),vec.end(),0);
    }
    int superPixel::getPixelNumberinUncertaintyArea() const {
        //smooth uncertainty map
        cv::Mat_<cv::Vec3b> gaussianBlurUncertainty;
        cv::Size kernel=cv::Size(5,5);
        cv::GaussianBlur(uncertainty,gaussianBlurUncertainty,kernel,5);
        int sum=0;
        cv::Size size=gaussianBlurUncertainty.size();
        for(auto r=0;r<size.height;r++)
            for(auto c=0;c<size.width;c++){
                int value=(int)gaussianBlurUncertainty(r,c)[0];
                if((float)value/(float)255>uncertaintyThreshold)
                    sum++;
            }
        return sum;
    }

    bool superPixel::testsuperpixeltouchanothersuperpixel(int index,std::vector<int>& newvector) {
        bool test=false;
#pragma omp parallel for
        for(const auto& pt:allsuperPixelCoordinates[index]){
            int t=std::max(0,pt.y-1);
            int l=std::max(0,pt.x-1);
            int r=std::min(pt.x+1,image.size().width);
            int b=std::min(pt.y+1,image.size().height);

            cv::Point top=cv::Point(pt.x,t),right=cv::Point(r,pt.y),left=cv::Point(l,pt.y),bot=cv::Point(pt.x,b);
#pragma  omp parallel for
            for(const auto& idx:newvector){
                for(const auto& p:allsuperPixelCoordinates[idx]){
                    if( top==p|| right==p|| left==p|| bot==p){
                        test=true;
                        break;
                    }
                }
            }
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(newvector.begin(),newvector.end(),gen);

        return test;
    }

    int superPixel::minRowwSuperPixel(std::vector<cv::Point> &superPixel) {
        std::vector<int>  vec;
        for(const auto& pt: superPixel){
            int vlaue=pt.y;
            vec.push_back(vlaue);
        }
        auto it=std::min_element(vec.begin(),vec.end());
        return *it;
    }
    void superPixel::createSuperPixelImage() {//create the super pixel image
        std::cout<<"start createSuperPixelImage***************"<<std::endl;
        cv::Mat labelmap,mask;
        int number;
        getSuperPixelImage(labelmap,mask,number);
        std::vector<int> row;
        cv::Mat R=labelmap.row(0).clone();
        for(auto i=0;i<R.cols;i++){
            row.push_back(R.at<int>(i));
        }
        std::sort(row.begin(),row.end());
        auto row1=std::unique(row.begin(),row.end());
        row.erase(row1,row.end());
         width=row.size();
        std::cout<<"width="<<width<<std::endl;
        std::vector<int> col;
        cv::Mat C=labelmap.col(0).clone();
        for(auto i=0;i<C.rows;i++)
            col.push_back(C.at<int>(i));
        std::sort(col.begin(),col.end());
        auto col1=std::unique(col.begin(),col.end());
        col.erase(col1,col.end());
        height=col.size();
        std::cout<<"height="<<height<<std::endl;
        std::cout<<"number="<<number<<std::endl;

        //CV_Assert(width*height==number);
        std::cout<<"select super pixel rate="<<(float)(width*height)/(float)number*100<<"%"<<std::endl;
        superpixelImage.create(height,width,CV_8U);
        for(auto r=0;r<height;r++){
            std::multimap<int,int> m;
            for(auto c=0;c<width;c++){
                std::vector<cv::Point> super;
                super=allsuperPixelCoordinates[r*width+c];
                int  value=miniColSuperPixel(super);
                m.insert(std::make_pair(value,r*width+c));
            }
            int c=0;
            for(auto it=m.begin();it!=m.end();it++,c++){
//                std::cout<<"m first="<<it->first<<";"<<it->second<<std::endl;
                LABEL l=preprocessSuperPixel(allsuperPixelCoordinates[it->second]);
                superpixelImage.at<uchar>(r,c)=l;
            }
        }
        std::cout<<"End************createSuperPixelImage*******"<<std::endl;
    }

    void superPixel::calculateLocalNeighborMap() {
        std::cout<<"start calculateLocalNeighborMap***************"<<std::endl;

        std::vector<cv::Mat> neighborsmap;
        neighborsmap.resize(ls.size());
        int kernel_size=3;
        cv::Mat_<uchar> kernel=cv::Mat_<uchar>::ones(kernel_size,kernel_size);
        //set the ankor point 0
        kernel.at<uchar>(kernel_size/2,kernel_size/2)=0;

        for(const auto& label:ls){
            cv::Mat superIMG=superpixelImage.clone();
            cv::Mat superPixelImg=cv::Mat::zeros(superIMG.size(),superIMG.type());
            for(auto r=0;r<superIMG.size().height;r++)
                for(auto c=0;c<superIMG.size().width;c++){
                    auto value=(LABEL)superIMG.at<uchar >(r,c);
                    if(value==label) superIMG.at<uchar >(r,c)=1;
                    else superIMG.at<uchar>(r,c)=0;
                }
            cv::filter2D(superIMG,neighborsmap[label],superIMG.depth(),kernel);
        }
        localSuperpixelNeighbormap.create(superpixelImage.size(),superpixelImage.type());

        for(auto r=0;r<localSuperpixelNeighbormap.size().height;r++)
            for(auto c=0;c<localSuperpixelNeighbormap.size().width;c++){
                LABEL trueLabel=superpixelImage.at<uchar>(r,c);
                std::vector<int> vec;
                for(const auto& label:ls){
                   //each point calcualte the maximum
                   vec.push_back(neighborsmap[label].at<uchar>(r,c));
                }
                //compare with the true label
                auto maxEle=std::max_element(vec.begin(),vec.end());
                if(*maxEle==trueLabel){
                    *maxEle=0;
                    maxEle=std::max_element(vec.begin(),vec.end());
                    LABEL predict=std::distance(vec.begin(),maxEle);
                    localSuperpixelNeighbormap.at<uchar >(r,c)=predict;
                }
                else{
                    LABEL predict=std::distance(vec.begin(),maxEle);
                    localSuperpixelNeighbormap.at<uchar >(r,c)=predict;
                }
        }


    }
#endif
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

        std::vector<int> allIndex(MaximumSuperPixelIndex,0);
        std::generate(allIndex.begin(), allIndex.end(), [n = 0]() mutable {return n++;});

        for(const auto& superpixel:allIndex){
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
//            vec[super_label]=0;
            //find the max
            auto maxEle=std::max_element(vec.begin(),vec.end());
            LABEL predict=std::distance(vec.begin(),maxEle);
//            std::for_each(vec.begin(),vec.end(),[&](int d){std::cout<<d;});
//            std::cout<<"true="<<super_label<<",predict="<<predict<<std::endl;
//            std::cout<<"all="<<allsuperPixelCoordinates.size()<<std::endl;
            int loc=superpixel;
//            std::cout<<"loc="<<loc<<std::endl;
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
            auto index = (LABEL) image.at<uchar>(pt);
            st[index] += 1;
        }
        auto it = std::max_element(st.begin(), st.end());
        LABEL label = std::distance(st.begin(),it);

        return label;
    }
#if 0
    int superPixel::extract_Contour(std::map<LABEL, std::vector<std::vector<cv::Point> > > &contours,
                                    std::list<std::vector<cv::Vec4i> > &hierarchies) {
        //convert 3 channel to 1 channel
        std::vector<cv::Mat> SrcMatpart(image.channels());
        cv::Mat img=convert_to_color(image);
        cv::split(img, SrcMatpart);
        //contour number
        int number = 0;
        //find the contour for different labels respectively,we can set the value as binary image, the label vale is 255 while other can be set to 0
        for (auto &label:ls) {
            cv::Mat temp;
            temp.create(SrcMatpart[0].size(), CV_8UC1);
            temp.setTo(0);
            //scan the whole image in one channel
//#pragma omp parallel for
            for (int r = 0; r < SrcMatpart[0].rows; r++)
//#pragma omp parallel for
                for (int c = 0; c < SrcMatpart[0].cols; c++) {
                    int value = (int) SrcMatpart[0].at<uchar>(r, c);
                    if (value == label) {
                        temp.at<uchar>(r, c) = 255;
                    }
                }
            std::vector<std::vector<cv::Point>> contour;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(temp, contour, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
            number += contour.size();

            contours.insert(std::make_pair(label, contour));
            hierarchies.emplace_front(hierarchy);
        }
        return number;
    }
#endif
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


                if (top != l and top != 255  and l>top) { NeighborMatrix(l, top)+=1; }
                if (left != l and left != 255  and l>left) { NeighborMatrix(l, left)+=1; }
                if (right != l and right != 255 and l>right) { NeighborMatrix(l, right)+=1; }
                if (bot != l and bot != 255 and l>bot) { NeighborMatrix(l, bot)+=1; }
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
        for(auto r=1;r<ext_labelMap.size().height-1;r++)
            for(auto c=1;c<ext_labelMap.size().width-1;c++){
                cv::Point top=cv::Point(c,r-1);
                cv::Point left=cv::Point(c-1,r);
                cv::Point right=cv::Point(c+1,r);
                cv::Point bot=cv::Point(c,r+1);
                int t=ext_labelMap.at<int>(top);
                int l=ext_labelMap.at<int>(left);
                int rg=ext_labelMap.at<int>(right);
                int b=ext_labelMap.at<int>(bot);
                int loc=ext_labelMap.at<int>(r,c);
                if(loc != t)  vec[loc].insert(t);
                if(loc != l)  vec[loc].insert(l);
                if(loc != rg) vec[loc].insert(rg);
                if(loc != b)  vec[loc].insert(b);

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
    }
#if 0
    bool superPixel::testIntersectionSuperPixel(const std::map<LABEL, std::vector<std::vector<cv::Point> > > &contours,
                                                const std::vector<cv::Point> &superPixels) {
        bool test = false;
        for (auto const &superpixel:superPixels) {
            auto it = contours.begin();
            for (; it != contours.end(); it++) {
                for (auto const &index:it->second)
                    if (testPointInPolygon(superpixel, index)) {
                        test = true;
                        break;
                    }
            }
        }
        return test;
    }

    bool superPixel::testIntersectionSuperPixel(const std::vector<cv::Point> &contour,
                                                const std::vector<cv::Point> &superPixels) {
        bool test=false;
        for(const auto& superpixel:superPixels){
            {
                if(testPointInPolygon(superpixel,contour)){
                    test=true;
                    break;
                }
            }
        }
        return test;
    }

    bool superPixel::testSuperPixelIntersectionReturnContour(
            const std::map<LABEL, std::vector<std::vector<cv::Point> > > &contours,
            const std::vector<cv::Point> &superPixel,
            std::multimap<LABEL, std::vector<cv::Point> > &result) {
        bool test = false;
        auto it = contours.begin();
//#pragma omp parallel for
        for (; it != contours.end(); it++) {
            int label = it->first;
//#pragma omp parallel for
            for (auto &index:it->second) {
//#pragma omp parallel for
                for (auto &superpixel:superPixel) {
                    if (testPointInPolygon(superpixel, index)) {
                        test = true;
                        result.insert(std::make_pair(label, index));
                        break;
                    }
                }
            }
        }
        return test;
    }

    bool superPixel::testSuperPixelIntersectionReturnContour(const std::vector<std::vector<cv::Point> > &contours,
                                                             const std::vector<cv::Point> &superPixels,
                                                             std::vector<std::vector<cv::Point>>  &result) {
        bool test=false;
        auto it = contours.begin();
#pragma omp parallel for
        for (; it != contours.end(); it++) {
#pragma omp parallel for
                for (auto &superpixel:superPixels) {
                    if (testPointInPolygon(superpixel, *it)) {
                        test = true;
                        result.emplace_back(superPixels);
                        break;
                    }
                }
        }
        return test;
    }
#endif
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

                    calculateSuperpixelRelationshipMap();
                    std::vector<int> newindexvector(allsuperPixelIntersectContour);
                    do{
                        std::vector<int> newvector;
                        std::vector<int> oldvector;
                        oldvector.clear();
                        oldvector.insert(oldvector.begin(),allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end());
                        //shuffle newindexvector
                        auto it=newindexvector.begin();
                        for(;it!=newindexvector.end();it++) {
                            allsuperPixelIntersectContour.insert(allsuperPixelIntersectContour.end(),
                                                                 superpixelRelationMap[*it].begin(),
                                                                 superpixelRelationMap[*it].end());
                            std::sort(allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end());
                            allsuperPixelIntersectContour.erase(std::unique(allsuperPixelIntersectContour.begin(),
                                                                            allsuperPixelIntersectContour.end()),
                                                                allsuperPixelIntersectContour.end());


//                            auto a4=superpixelRelationMap[*it][0];
                            newvector.insert(newvector.begin(), superpixelRelationMap[*it].begin(),
                                             superpixelRelationMap[*it].end());

                        }
                        superpixelonboundarysize=N-allsuperPixelIntersectContour.size();
                        if (superpixelonboundarysize == 0){
                            std::copy(allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end(),
                                      selectedSuperPixelIndex.begin());
                            break;
                        }
                        if(superpixelonboundarysize<0){
                            selectedSuperPixelIndex.insert(selectedSuperPixelIndex.begin(),
                                                           oldvector.begin(),oldvector.end());
                            std::vector<int> rest;
                            std::set_difference(allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end(),
                                                oldvector.begin(),oldvector.end(),std::inserter(rest,rest.begin()));

                            std::random_device RD;
                            std::mt19937 GEN(RD());
                            std::shuffle(rest.begin(),rest.end(),GEN);
                            auto it1=rest.begin();
                            auto len=N-oldvector.size();
                            selectedSuperPixelIndex.insert(selectedSuperPixelIndex.end(),it1,it1+len);
                            break;
                        }

                        newvector.clear();
                        newindexvector.clear();
                        newindexvector.resize(newvector.size());
                        std::copy(newvector.begin(),newvector.end(),newindexvector.begin());

                    }while(superpixelonboundarysize>0);
#if 0
                    int rest=N-superpixelonboundarysize;//need find rest super-pixel more

                    std::vector<int> unselected;
                    std::set_difference(vec.begin(), vec.end(), allsuperPixelIntersectContour.begin(), allsuperPixelIntersectContour.end(),
                                        std::inserter(unselected, unselected.begin()));
//                        std::for_each(unselected.begin(),unselected.end(),[](int b){
//                            std::cout<<"new="<<b<<std::endl;
//                        });
                    std::vector<int> newindexvector(allsuperPixelIntersectContour.size());
                    std::copy(allsuperPixelIntersectContour.begin(),allsuperPixelIntersectContour.end(),newindexvector.begin());

//                        std::for_each(newindexvector.begin(),newindexvector.end(),[](int b){
//                            std::cout<<"new="<<b<<std::endl;
//                        });

                    do{
                        std::vector<int> newvector;
                        ////
                        auto it=unselected.begin();
                        for(;it!=unselected.end();it++){
                            if(testsuperpixeltouchanothersuperpixel(*it,newindexvector)){
                                selectedSuperPixelIndex.push_back(*it);
                                newvector.push_back(*it);
                                rest--;
                                if(rest<0)
                                    break;
                            }
                        }////////
                        std::vector<int> vec1;
                        std::set_difference(unselected.begin(), unselected.end(), newvector.begin(), newvector.end(),
                                            std::inserter(vec1, vec1.begin()));
                        unselected.clear();
                        unselected.resize(vec1.size());
                        std::copy(vec1.begin(),vec1.end(),unselected.begin());

                        newindexvector.clear();
                        newindexvector.resize(newvector.size());
                        std::copy(newvector.begin(),newvector.end(),newindexvector.begin());
                    }while(rest>0);
#endif
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
//                        std::cout<<"label="<<label<<std::endl;
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

//                std::cout << "size_superpixel=" << selectedSuperPixel.size() << std::endl;
                cv::Mat_<float> confusionmatrix;
                confusionmatrix.create(confusionMatrix.size());
                confusionMatrix.copyTo(confusionmatrix);
                // the true label can be most set by the label based on the row

                //set the diag value 0
                for(int r = 0; r < confusionmatrix.size().height; r++)
                    for (int c = 0; c < confusionmatrix.size().width; c++)
                    {
                        if(r==c)
                            confusionmatrix(r,c)=0.f;
                    }
                //find the most confused label
                std::map<int, int> confusedMap;//the first is the true label the sencond is the confused label
                for (int r = 0; r < confusionmatrix.size().height; r++) {
                    cv::Point  maxLoc;
                    cv::Mat row=confusionmatrix.row(r);
                    cv::minMaxLoc(row, nullptr, nullptr, nullptr,&maxLoc);
                    confusedMap.insert(std::make_pair(r,maxLoc.x));
                }

                if (confusedMap.empty())
                    return;
                else {
                    for (const auto &m:confusedMap)
                        for (const auto &sp:selectedSuperPixelIndex) {
                            auto label=preprocessSuperPixel(allsuperPixelCoordinates[sp]);
                            if(m.first==label){
                                for (const auto &pt:allsuperPixelCoordinates[sp]) {
                                    addedLabelNoiseImage.at<cv::Vec3b>(pt)[0] = m.second;
                                    addedLabelNoiseImage.at<cv::Vec3b>(pt)[1] = m.second;
                                    addedLabelNoiseImage.at<cv::Vec3b>(pt)[2] = m.second;
                                }
                            }
                        }
                }

                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END CONFUSION RELABEL***************"<<std::endl;

                break;
            }

            case MostNeighbour: {//how often(GLOBAL)
                std::cout<<"*************************START MOST NEIGHBOUR RELABEL***************"<<std::endl;
                clock_t startTime,endTime;
                startTime = clock();

                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);

                std::cout<<"NeighborMatrix="<<NeighborMatrix<<std::endl;
                cv::Point point;
                cv::minMaxLoc(NeighborMatrix, nullptr, nullptr, nullptr, &point);

                LABEL relabel;
                LABEL mostConfusedlabel1=point.x,mostConfusedlabel2=point.y;

                for (const auto &superpixel:selectedSuperPixelIndex) {
                    LABEL l = preprocessSuperPixel(allsuperPixelCoordinates[superpixel]);
                    if (l == mostConfusedlabel1) {
                        relabel=mostConfusedlabel2;
                        for (const auto &pt:allsuperPixelCoordinates[superpixel]) {
                            addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(relabel,relabel,relabel);
                        }
                    }
                    if (l == mostConfusedlabel2) {
                        relabel=mostConfusedlabel1;
                        for (const auto &pt:allsuperPixelCoordinates[superpixel]) {
                            addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(relabel,relabel,relabel);
                        }
                    }
                }
                endTime = clock();
                std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END MOST NEIGHBOUR RELABEL***************"<<std::endl;

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

                for(auto u=0;u<size.width;u++)
                    for(auto v=0;v<size.height;v++)
                    {
                        cv::Point p=cv::Point(u,v);
                        if(selectedMap(p)==0){
                            cv::Mat features_pixel;
                            for(auto pr=0;pr<patch_size;pr++)
                                for(auto pc=0;pc<patch_size;pc++){
                                    int x=u+pc-band_size;
                                    int y=v+pr-band_size;
                                    //limit the region
                                    if(x<0) x=0;
                                    if(y<0) y=0;
                                    if(x>size.width) x=size.width-1;
                                    if(y>size.height) y=size.height-1;

                                    features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x,y))[0]);
                                    features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x,y))[1]);
                                    features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x,y))[2]);
                                }
                            features_pixel.convertTo(features_pixel, CV_32F);
                            trainData.push_back(features_pixel.reshape(1,1));
                            trainLabels.push_back((LABEL) image.at<uchar>(v,u));
                        }
                    }
                std::cout<<"*************************START KNN TRAIN ***************"<<std::endl;
                endtime1=clock();
                cv::Ptr<cv::ml::KNearest> knn=cv::ml::KNearest::create();
                knn->train(trainData,cv::ml::ROW_SAMPLE,trainLabels);
                endtime2=clock();
                std::cout << "Totle Time : " <<(double)(endtime2 - endtime1) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END KNN TRAIN *****************"<<std::endl;

                std::cout<<"*************************START CREATE NEAREST MAP ***************"<<std::endl;
                endtime3=clock();
                //create nearest map
                cv::Mat_<float> nearestMap=cv::Mat_<float>::zeros(image.size());
                int num = size.width*size.height;//2 dim2---> 1 dim for openmp
#pragma omp parallel for
                for (int i = 0;i<num;i++)
                {
                    int u=i%size.width;
                    int v=i/size.width;
                    cv::Point p=cv::Point(u,v);
                    if(selectedMap(p)==1){
                        cv::Mat features_pixel;
                        for (auto pr = 0; pr < patch_size; pr++)
                            for (auto pc = 0; pc < patch_size; pc++) {
                                int x = u + pc - band_size;
                                int y = v + pr - band_size;
                                //limit the region
                                if (x < 0) x = 0;
                                if (y < 0) y = 0;
                                if (x > size.width) x = size.width - 1;
                                if (y > size.height) y = size.height - 1;

                                features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x, y))[0]);
                                features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x, y))[1]);
                                features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x, y))[2]);
                                }
                            features_pixel.convertTo(features_pixel, CV_32F);
                            cv::Mat testData;
                            testData.push_back(features_pixel.reshape(1, 1));
                            cv::Mat predictedLabel;
                            int K = 51;
                            knn->findNearest(testData, K, predictedLabel);
                            nearestMap(p)=predictedLabel.at<float>(0);
                        }
                    }

                endtime4=clock();
                std::cout << "Totle Time : " <<(double)(endtime4 - endtime3) / CLOCKS_PER_SEC << "s" << std::endl;
                std::cout<<"*************************END CREATE NEAREST MAP ***************"<<std::endl;

                //find the majority for each superpixel
                for(const auto&superpixel:selectedSuperPixelIndex) {
                    std::vector<int> labels;
                    labels.resize(ls.size(),0);
                    for (const auto &pt:allsuperPixelCoordinates[superpixel]) {
                        LABEL l=(LABEL)nearestMap(pt);
                        labels[l]+=1;
                    }
                    LABEL relabel=std::max_element(labels.begin(),labels.end())-labels.begin();
                    LABEL truelabel=preprocessSuperPixel(allsuperPixelCoordinates[superpixel]);

                    if(relabel==truelabel){
                        auto it=std::max_element(labels.begin(),labels.end());
                        *it=0;
                        relabel=std::max_element(labels.begin(),labels.end())-labels.begin();
                    }

                    for(const auto& pt:allsuperPixelCoordinates[superpixel]){
                        addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b(relabel,relabel,relabel);
                    }
                }

#if 0
                for(auto k=0;k<selectedSuperPixelIndex.size();k++) {
                    //calculate the majority
                    std::vector<LABEL> labels;
                    for (auto j=0;j<allsuperPixelCoordinates[selectedSuperPixelIndex[k]].size();j++) {
                        int u =  allsuperPixelCoordinates[selectedSuperPixelIndex[k]][j].x;
                        int v =  allsuperPixelCoordinates[selectedSuperPixelIndex[k]][j].y;
                        cv::Mat features_pixel;
                        for (auto pr = 0; pr < patch_size; pr++)
                            for (auto pc = 0; pc < patch_size; pc++) {
                                int x = u + pc - band_size;
                                int y = v + pr - band_size;
                                //limit the region
                                if (x < 0) x = 0;
                                if (y < 0) y = 0;
                                if (x > size.width) x = size.width - 1;
                                if (y > size.height) y = size.height - 1;

                                features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x, y))[0]);
                                features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x, y))[1]);
                                features_pixel.push_back(featureImg.at<cv::Vec3b>(cv::Point(x, y))[2]);
                            }
                        features_pixel.convertTo(features_pixel, CV_32F);
                        cv::Mat testData;
                        testData.push_back(features_pixel.reshape(1, 1));

                        cv::Mat predictedLabel, neighbours;
                        int K = 51;
                        knn->findNearest(testData, K, predictedLabel, neighbours);
                        labels.push_back(predictedLabel.at<float>(0));
                    }

                    //compute the majority
                    std::vector<LABEL> l;
                    l.resize(ls.size());

                    for(const auto&label:ls){//count the number of each class
                        l[label]=std::count(labels.begin(),labels.end(),label);
                    }
                    LABEL relabel=std::max_element(l.begin(),l.end())-l.begin();

                    LABEL truelabel=preprocessSuperPixel(allsuperPixelCoordinates[selectedSuperPixelIndex[k]]);
                    //exclude the same label and different label
                    if(relabel==truelabel){
                        auto it=std::max_element(l.begin(),l.end());
                        *it=0;
                        relabel=std::max_element(l.begin(),l.end())-l.begin();
                    }

                    for(const auto& pt:allsuperPixelCoordinates[selectedSuperPixelIndex[k]]){
                        addedLabelNoiseImage.at<cv::Vec3b>(pt)=cv::Vec3b(relabel,relabel,relabel);
                    }
                }
#endif
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
            case RandomSelectMostNeighbourRelabel:{
                //**********Select method:random selection********************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::MostNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectMostNeighbourRelabel"+outputFilenameIndex);
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
            case UncertaintyAreaSelectMostNeighbourRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::MostNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintyAreaSelectMostNeighbourRelabel"+outputFilenameIndex);
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

            case ObjectBorderSelectMostNeighbourRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::MostNeighbour);
                //process the selection
                sampleSelection(N);
                //process the add label
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectMostNeighbourRelabel"+outputFilenameIndex);
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

                //most neighor
                setAddLabelNoiseType(SuperPixel::NewLabelType::MostNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("RandomSelectMostNeighbourRelabel"+outputFilenameIndex);

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
                saveColorImage("UncertaintySelectRandomRelabel"+outputFilenameIndex);

                //confusion relabel
                setAddLabelNoiseType(SuperPixel::NewLabelType::CMrelabel);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintySelectConfusionRelabel"+outputFilenameIndex);

                //most neighor
                setAddLabelNoiseType(SuperPixel::NewLabelType::MostNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintySelectMostNeighbourRelabel"+outputFilenameIndex);

                //nearest neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::NearestNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintySelectNearestNeighbourRelabel"+outputFilenameIndex);

                //local neighbor
                setAddLabelNoiseType(SuperPixel::NewLabelType::localNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("UncertaintySelectLocalNeighbourRelabel"+outputFilenameIndex);

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

                //most neighor
                setAddLabelNoiseType(SuperPixel::NewLabelType::MostNeighbour);
                addLabelNoise();
                //save color image
                saveColorImage("ObjectBorderSelectMostNeighbourRelabel"+outputFilenameIndex);

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