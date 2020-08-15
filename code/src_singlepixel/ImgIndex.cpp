

#include "ImgIndex.h"

ImgIndex::ImgIndex() {
    this->iPos = 0; //image index
    this->xPos = 0; // column number of pixel
    this->yPos = 0; // row number of corresponding pixel
}

ImgIndex::ImgIndex(int i, int x, int y) {
    this->iPos = i;
    this->xPos = x;
    this->yPos = y;
}

ImgIndex::~ImgIndex() = default;


int ImgIndex::getI() {
    return iPos;
}

int ImgIndex::getX() {
    return xPos;
}

int ImgIndex::getY() {
    return yPos;
}
