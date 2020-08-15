
// define different types of filters to reduce feature noise
// define different types of pixel selection and pixel relabelling approaches

#ifndef HTCV_ENUM_H
#define HTCV_ENUM_H

enum filterTypes {bilateral_filter, NLM_filter, BM3D_filter};
enum PixelSelectionType{randomSelection, uncertaintyArea, objectBorder};
//enum RelabelType{singlePixel, superPixel};
enum NewLabelType{randomRelabel, CMrelabel, localNeighbourInReference, globalNeighbourInReference, neighbourInFeatureSpace};
#endif //HTCV_ENUM_H
