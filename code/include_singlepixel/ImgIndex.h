
#ifndef HTCV_IMGINDEX_H
#define HTCV_IMGINDEX_H


#include <iostream>

using namespace std;

class ImgIndex {
public:
    // i:image index; x: column number of pixel; y: row number of pixel
    ImgIndex();

    ImgIndex(int i, int x, int y);

    ~ImgIndex();

    int getI();

    int getX();

    int getY();

private:
    int iPos; //index of image
    int xPos; // spatial coordinate of pixel
    int yPos;
};


#endif //HTCV_IMGINDEX_H
