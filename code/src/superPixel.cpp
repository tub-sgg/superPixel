//
// Created by alxlee on 05.07.20.
//

#include "superPixel.h"

namespace SuperPixel {
//onstruction


    cv::Mat superPixel::convert_from_color(const cv::Mat &src) {
        cv::Mat label_map = cv::Mat::zeros(src.size(), src.type());
        for (int i = 0; i < label_map.size().height; i++) {
            for (int j = 0; j < label_map.size().width; j++) {
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 0, 0))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(1, 1, 1);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 0))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(2, 2, 2);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 255, 0))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(3, 3, 3);
                if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 255, 255))
                    label_map.at<cv::Vec3b>(i, j) = cv::Vec3b(4, 4, 4);
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

//return the label of majority pixel in superpixls
    LABEL superPixel::preprocessSuperPixel(const std::vector<cv::Point> &superpixel) {
        const int labelNumber = ls.size();
        std::vector<int> st(labelNumber, 0);
        for (const auto &pt:superpixel) {
            LABEL index = (LABEL) image.at<uchar>(pt);
            st[index] += 1;
        }

        std::vector<LABEL>::iterator it;
        it = std::max_element(st.begin(), st.end());
        LABEL label = std::distance(st.begin(),it);

        return label;
    }

    int superPixel::extract_Contour(std::map<LABEL, std::vector<std::vector<cv::Point> > > &contours,
                                    std::list<std::vector<cv::Vec4i> > &hierarchies) {
        //convert 3 channel to 1 channel
        std::vector<cv::Mat> SrcMatpart(image.channels());
        cv::split(image, SrcMatpart);
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

    void superPixel::extract_Contour(std::vector<std::vector<cv::Point> > &contours) {
        //convert 3 channel to 1 channel
        std::vector<cv::Mat> SrcMatpart(image.channels());
        cv::split(image, SrcMatpart);
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(SrcMatpart[0], contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
    }

    bool superPixel::testPointInPolygon(const cv::Point &point, const std::vector<cv::Point> &contour) {
        boost::geometry::model::polygon<pointXY> ply;
        for (auto const &points:contour) {
            boost::geometry::append(ply, point_t(points.x, points.y));
        }
        //close the polygon
        boost::geometry::append(ply, point_t(contour[0].x, contour[0].y));

        return boost::geometry::within(point_t(point.x, point.y), ply);
    }

    void superPixel::calculateNeighborNumber(cv::Mat_<int> &NeighborMatrix) {
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

                int l = (int) ext_labelMap.at<cv::Vec3b>(r, c)[0];

                if (top != l and top != 255 and l > top) { NeighborMatrix(l, top)++; }
                if (left != l and left != 255 and l > left) { NeighborMatrix(l, left)++; }
                if (right != l and right != 255 and l > right) { NeighborMatrix(l, right)++; }
                if (bot != l and bot != 255 and l > bot) { NeighborMatrix(l, bot)++; }
            }
    }

    void superPixel::getSuperPixelImage(cv::Mat &labelImage, cv::Mat &mask, int &labelNumbers) {
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(image,
                                                                                        cv::ximgproc::MSLIC,
                                                                                        superPixelSize);
        slic->iterate();
        slic->enforceLabelConnectivity();
        slic->getLabelContourMask(mask);
        slic->getLabels(labelImage);
        labelNumbers = slic->getNumberOfSuperpixels();
    }

    int superPixel::getSuperPixelCoordinates(std::vector<std::vector<cv::Point> > &superPixels) {
        cv::Mat labelImage;
        cv::Mat mask;
        int labelNumber;
        getSuperPixelImage(labelImage, mask, labelNumber);
        for (int index = 0; index <= labelNumber; index++) {
            std::vector<cv::Point> point;
#pragma omp parallel for
            for (int r = 0; r < labelImage.size().height; r++)
#pragma omp parallel for
                for (int c = 0; c < labelImage.size().width; c++) {
                    int value = labelImage.at<int>(r, c);
                    if (index == value) {
                        point.emplace_back(cv::Point(c, r));
                    }
                }
            superPixels.emplace_back(point);
        }
        return labelNumber;//how many superpixel generated
    }


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
    bool superPixel::testIntersectionSuperPixel(const std::vector<std::vector<cv::Point> > &contours,
                                                const std::vector<cv::Point> &superPixels) {
        bool test=false;
        for(const auto& superpixel:superPixels){
            for(const auto& index:contours){
                if(testPointInPolygon(superpixel,index)){
                    test=true;
                    break;
                }
            }
        }
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

    void superPixel::sampleSelection() {
        switch (selectionType) {
            case randomSelection: {

                std::vector<std::vector<cv::Point>> superPixels;
                int maxmumSuperPixelNumber = getSuperPixelCoordinates(superPixels);
                int randomNumber = static_cast<int>(maxmumSuperPixelNumber * randomRate);
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                std::shuffle(superPixels.begin(), superPixels.end(), std::default_random_engine(seed));

                auto it = superPixels.begin();
                selectedSuperPixel.insert(selectedSuperPixel.begin(), it, it + randomNumber);
                break;
            }

            case uncertaintyArea: {
                //average threshold
                std::vector<std::vector<cv::Point> > superPixels;
                getSuperPixelCoordinates(superPixels);
                for (const auto &superPixel:superPixels) {
                    float value = 0.f;
                    for (const auto &pt:superPixel) {
                        value += (float) uncertainty.at<uchar>(pt) / (float) 255;
                    }

                    if (value / superPixel.size() > uncertaintyThreshold) {
                        selectedSuperPixel.insert(selectedSuperPixel.begin(), superPixel);
                    }
                }
                std::cout<<"end.......uncertainty Area"<<std::endl;
                break;
            }

            case objectBorder: {
                std::vector<std::vector<cv::Point>> contours;
                extract_Contour(contours);

                std::vector<std::vector<cv::Point>> allSuperPixels;
                getSuperPixelCoordinates(allSuperPixels);

                for (auto const &sp:allSuperPixels) {
                    if (testIntersectionSuperPixel(contours, sp)) {
                        selectedSuperPixel.emplace_back(sp);
                    }
                }
                std::cout<<"end.......object boundary"<<std::endl;
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
                CV_Assert(image.channels() == 3);

                addedLabelNoiseImage.create(image.size(), image.type());
                this->image.copyTo(addedLabelNoiseImage);
                std::cout<<"random_superpixel size="<<this->addedLabelNoiseImage.size()<<std::endl;
                //copy label list
                std::vector<LABEL> l;
                for (auto it:ls)
                    l.push_back(it);

                for (auto const &sp:selectedSuperPixel) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::shuffle(l.begin(), l.end(), gen);
                    LABEL label = l[0];
                    for (auto const &pt:sp) {
//                        std::cout<<"label="<<label<<std::endl;
                        addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(label, label, label);
                    }
                }
                std::cout << l[0] << std::endl;
                break;
            }

            case CMrelabel: {//confusionMatrix
                CV_Assert(image.channels() == 3);
                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);

                std::cout << "size_superpixel=" << selectedSuperPixel.size() << std::endl;
                cv::Mat_<float> confusionmatrix;
                confusionmatrix.create(confusionMatrix.size());
                confusionMatrix.copyTo(confusionmatrix);
                // the true label can be most set by the label based on the row

                //find the most confused label
                std::map<int, int> confusedMap;//the first is the true label the sencond is the confused label
                for (int r = 0; r < confusionmatrix.size().height; r++) {
                    float s = 0.f;
                    int h = 0;
                    int w = 0;
                    for (int c = 0; c < confusionmatrix.size().width; c++) {
                        if(r==c){confusionmatrix(r,c)=0.f;}
                        float value = confusionmatrix(r, c);
                        if (value > s) {
                            s = value;
                            h = r;
                            w = c;
                        }
                    }
                    if (h != w)
                        confusedMap.insert(std::make_pair(h, w));
                }
                if (confusedMap.empty())
                    return;
                else {
                    for (const auto &m:confusedMap)
                        for (const auto &sp:selectedSuperPixel) {
                            for (const auto &pt:sp) {
                                LABEL label = (LABEL) image.at<cv::Vec3b>(pt)[0];
                                if (label == m.second) {
                                    addedLabelNoiseImage.at<cv::Vec3b>(pt)[0] = m.first;
                                    addedLabelNoiseImage.at<cv::Vec3b>(pt)[1] = m.first;
                                    addedLabelNoiseImage.at<cv::Vec3b>(pt)[2] = m.first;
                                }
                            }
                        }
                }
                break;
            }

            case neighbourInReference: {//how often
                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);

                cv::Mat_<int> NeighborMatrix;
                calculateNeighborNumber(NeighborMatrix);

//                std::cout<<"NeighborMatrix="<<NeighborMatrix<<std::endl;
                cv::Point point;
                cv::minMaxLoc(NeighborMatrix, nullptr, nullptr, nullptr, &point);

                LABEL label = point.y;
                LABEL relabel = point.x;

                for (const auto &superpixel:selectedSuperPixel) {
                    LABEL l = preprocessSuperPixel(superpixel);
                    if (l == label) {
                        l = relabel;
                        for (const auto &pt:superpixel) {
                            addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(l, l, l);
                        }
                    }
                }
                break;
            }
            case neighbourInFeatureSpace: {//nearest neighbor==intersection
                addedLabelNoiseImage.create(image.size(), image.type());
                image.copyTo(addedLabelNoiseImage);
                std::vector<std::vector<cv::Point> >  contours;
                extract_Contour(contours);
#pragma omp parallel for
                for (const auto &superpixel:selectedSuperPixel) {
                    if(testIntersectionSuperPixel(contours,superpixel)){
                        auto l = (LABEL)image.at<uchar>(contours[0][0]);//casually select label value,select the first
                        for (const auto &pt:superpixel) {
                            addedLabelNoiseImage.at<cv::Vec3b>(pt) = cv::Vec3b(l, l, l);
                        }
                    }
                }
                std::cout<<"end..neighbourInFeatureSpace"<<std::endl;
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

    void superPixel::superPixelNoise(SuperPixel::options option) {
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
                sampleSelection();
                //process the add label
                addLabelNoise();
                break;
            }
            case RandomSelectNeighbourInReferenceRelabel:{
                //**********Select method:random selection********************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::neighbourInReference);
                //process the selection
                sampleSelection();
                //process the add label
                addLabelNoise();
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
                sampleSelection();
                //process the add label
                addLabelNoise();
                break;
            }

            case RandomSelectNeighbourInFeatureSpaceRelabel:{
                //**********Select method:random selection********************
                //**********Relabel method:confusion selection****************
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::randomSelection);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::neighbourInReference);
                //process the selection
                sampleSelection();
                //process the add label
                addLabelNoise();
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
                sampleSelection();
                //process the add label
                addLabelNoise();
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
                sampleSelection();
                //process the add label
                addLabelNoise();
                break;
            }
            case UncertaintyAreaSelectNeighbourInReferenceRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::neighbourInReference);
                //process the selection
                sampleSelection();
                //process the add label
                addLabelNoise();
                break;
            }
            case UncertaintyAreaSelectNeighbourInFeatureSpaceRelabel:{
                //**********Select method:Uncertainty selection***************
                //**********Relabel method:neighbourInFeatureSpace selection**
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::uncertaintyArea);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::neighbourInFeatureSpace);
                //process the selection
                sampleSelection();
                //process the add label
                addLabelNoise();
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
                sampleSelection();
                //process the add label
                addLabelNoise();
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
                sampleSelection();
                //process the add label
                addLabelNoise();
                break;
            }

            case ObjectBorderSelectNeighbourInReferenceRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:neighbourInReference selection*****
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::neighbourInReference);
                //process the selection
                sampleSelection();
                //process the add label
                addLabelNoise();
                break;
            }
            case ObjectBorderSelectNeighbourInFeatureSpaceRelabel:{
                //**********Select method:objectBorder selection**************
                //**********Relabel method:neighbourInFeatureSpace selection**
                //************************************************************
                //select the section type
                setPixelSelectionType(SuperPixel::PixelSelectionType::objectBorder);
                //select the relabel type
                setAddLabelNoiseType(SuperPixel::NewLabelType::neighbourInFeatureSpace);
                //process the selection
                sampleSelection();
                //process the add label
                addLabelNoise();
                break;
            }
            default:{
                std::cerr<<"error option selection"<<std::endl;
            }

        }

    }


}