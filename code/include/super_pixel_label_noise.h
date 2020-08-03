//
// Created by alxlee on 08.06.20.
//

#ifndef NOISE_SUPER_PIXEL_LABEL_NOISE_H
#define NOISE_SUPER_PIXEL_LABEL_NOISE_H

namespace SuperPixel {
    enum options {
        RandomSelectRandomRelabel,
        RandomSelectConfusionRelabel,
        RandomSelectMostNeighbourRelabel,
        RandomSelectNearestNeighbourRelabel,
        RandomSelectLocalNeighbourRelabel,

        UncertaintyAreaSelectRandomRelabel,
        UncertaintyAreaSelectConfusionRelabel,
        UncertaintyAreaSelectMostNeighbourRelabel,
        UncertaintyAreaSelectNearestNeighbourRelabel,
        UncertaintyAreaSelectLocalNeighbourRelabel,

        ObjectBorderSelectRandomRelabel,
        ObjectBorderSelectConfusionRelabel,
        ObjectBorderSelectMostNeighbourRelabel,
        ObjectBorderSelectNearestNeighbourRelabel,
        ObjectBorderSelectLocalNeighbourRelabel,

        RandomSelectionAllRelabel,
        UncertaintySelectAllRelabel,
        ObjectBorderSelectAllRelabel

    };

    enum PixelSelectionType {
        randomSelection, uncertaintyArea, objectBorder
    };
    enum NewLabelType {
        randomRelabel, CMrelabel, MostNeighbour, localNeighbour,NearestNeighbour
    };

}
#endif //NOISE_SUPER_PIXEL_LABEL_NOISE_H
